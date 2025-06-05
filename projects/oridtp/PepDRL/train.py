from __future__ import print_function
import argparse
import random
from collections import deque
from affinity_env import Seq_env, Mutate
from mcts_mutate import MCTSMutater
from pvnet import PolicyValueNet
import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()
def initialize_weights(peptide_length):
    """Initialize weights and generate a random peptide sequence."""
    weights = np.random.gumbel(0, 1, (peptide_length, 20))
    weights = np.array([np.exp(weights[i]) / np.sum(np.exp(weights[i])) for i in range(len(weights))])
    amino_acids = np.array(['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'])
    peptide_sequence = ''.join(amino_acids[np.argmax(weights, axis=1)])
    return weights, peptide_sequence

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

class TrainingPipeline:
    def __init__(self, pdb_id, dual_pdbid, initial_sequence, num_playouts, batch_size, alphabet, num_iterations, c_puct, jumpout, initial_model=None):
        self.pdb_id = pdb_id
        self.dual_pdbid = dual_pdbid
        self.sequence_length = len(initial_sequence)
        self.vocabulary_size = len(alphabet)
        self.environment = Seq_env(self.sequence_length, alphabet, self.pdb_id, self.dual_pdbid, initial_sequence)
        self.mutation_strategy = Mutate(self.environment)
        self.learning_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temperature = 1.0
        self.num_playouts = num_playouts
        self.c_puct = c_puct
        self.jumpout = jumpout
        self.buffer_size = num_iterations
        self.batch_size = batch_size
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_target = 0.02
        self.check_frequency = 50
        self.game_batch_count = num_iterations
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_count = 1000
        self.buffer_no_extend = False
        self.update_predictor = 0
        self.predictor_index_set = set()
        self.collected_sequences = set()
        self.sequences_and_fitness = []
        self.generated_peptides = []
        self.loss_history = []
        self.play_sequence_history = []
        self.play_loss_history = []
        self.policy_dict = {}
        self.mutation_policy_dict = {}
        self.loss_A_list = []
        self.loss_B_list = []

        if initial_model:
            self.policy_value_net = PolicyValueNet(self.sequence_length, self.vocabulary_size, model_file=initial_model, use_gpu=True)
        else:
            self.policy_value_net = PolicyValueNet(self.sequence_length, self.vocabulary_size, use_gpu=True)
        self.mcts_player = MCTSMutater(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.num_playouts, is_selfplay=1)

    def collect_selfplay_data(self, num_games=1):
        """Collect data from self-play games."""
        self.update_predictor = 0
        self.buffer_no_extend = False
        for _ in range(num_games):
            play_data, peptides, play_sequences, play_losses, policy_dict = self.mutation_strategy.start_mutating(self.mcts_player)
            play_data = list(play_data)[:]
            self.episode_length = len(play_data)
            self.policy_dict = policy_dict
            self.mutation_policy_dict.update(self.policy_dict)
            if self.episode_length == 0:
                self.buffer_no_extend = True
            else:
                self.data_buffer.extend(play_data)
                for peptide, loss, loss_A, loss_B in peptides:
                    if peptide not in self.generated_peptides:
                        self.generated_peptides.append(peptide)
                        self.loss_history.append(loss)
                        self.loss_A_list.append(loss_A)
                        self.loss_B_list.append(loss_B)


                    if peptide not in self.mutation_policy_dict.keys():
                        self.mutation_policy_dict[peptide] = [loss, loss_A, loss_B]

                self.play_sequence_history.extend(play_sequences)
                self.play_loss_history.extend(play_losses)

    def policy_update(self):
        """Update the policy using collected data."""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_value = self.policy_value_net.policy_value(state_batch)
        
        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learning_rate * self.lr_multiplier)
            new_probs, new_value = self.policy_value_net.policy_value(state_batch)
            kl_divergence = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl_divergence > self.kl_target * 4:
                break
        
        if kl_divergence > self.kl_target * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl_divergence < self.kl_target / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        return loss, entropy

    def run(self, output_directory="./output/init0/"):
        """Run the training pipeline."""
        try:
            for i in range(self.game_batch_count):
                print("######### {}-th Play ###########".format(i + 1))
                self.collect_selfplay_data(self.play_batch_size)
                

                df_generated = pd.DataFrame({"peptide": self.generated_peptides, "loss": self.loss_history})
                # df_generated.to_csv(f"{output_directory}/generated_peptides.csv", index=False)

                df_play = pd.DataFrame({"play_peptide": self.play_sequence_history, "play_loss": self.play_loss_history})
                # df_play.to_csv(f"{output_directory}/play_data.csv", index=False)

                policy_losses = np.array(list(self.policy_dict.values()))[:, 0]
                policy_fitness = 1 / (policy_losses * 1000)
                policy_sequences = np.array(list(self.policy_dict.keys()))
                df_policy = pd.DataFrame({"peptide": policy_sequences, "loss": policy_fitness})
                # df_policy.to_csv(f"{output_directory}/unique_playout.csv", index=False)

                mutation_policy_values = list(self.mutation_policy_dict.values())
                max_length = max(len(x) for x in mutation_policy_values)
                padded_mutation_policy_values = [x + [0] * (max_length - len(x)) for x in mutation_policy_values]
                mutation_policy_losses = np.array(padded_mutation_policy_values)[:, 0]
                mutation_policy_fitness = 1 / (mutation_policy_losses * 1000)

                losses_A = np.array(mutation_policy_values)[:, 1]
                losses_B = np.array(mutation_policy_values)[:, 2]


                mutation_policy_sequences = np.array(list(self.mutation_policy_dict.keys()))
                df_mutation_policy = pd.DataFrame({"peptide": mutation_policy_sequences, "loss": mutation_policy_fitness, "losses_A": losses_A, "losses_B": losses_B})
                df_mutation_policy.to_csv(f"{output_directory}/Move_and_playout.csv", index=False)

                if len(self.data_buffer) > self.batch_size and not self.buffer_no_extend:
                    loss, entropy = self.policy_update()

        except KeyboardInterrupt:
            print('\n\rTraining interrupted.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCTS_Evobind')
    parser.add_argument("--output_dir", type=str, default="/alldata/LChuang_data/workspace/Project/Protein_design/MCTS_evobind/output/", help="Output data directory")
    parser.add_argument("--start_sequence", type=str, default=None, help="Starting peptide sequence")
    parser.add_argument("--pdbid", type=str, default='1ssc', help="PDB ID")
    parser.add_argument("--dual_pdbid", type=str, default='1ssc', help="DUAL PDB ID")
    parser.add_argument("--n_playout", type=int, default=11, help="Number of playouts")
    parser.add_argument("--init_num", type=str, default='1', help="Initialization seed")
    parser.add_argument("--batch_size", type=int, default=32, help="MCTS batch size")
    parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--jumpout", type=int, default=1000, help="Jumpout parameter (1000 for no jump)")
    parser.add_argument("--c_puct", type=float, default=3, help="Exploration parameter for MCTS")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep time to avoid resource exhaustion")
    parser.add_argument("--length", type=int, help="Length of the peptide sequence")
    args = parser.parse_args()

    if int(args.sleep) > 0:
        sleep_time = args.sleep * 5
        time.sleep(sleep_time)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    start_time = datetime.now()
    print("The current date and time is", start_time)

    init_num_list = args.init_num.split(',')
    num_playouts = args.n_playout
    base_output_directory = args.output_dir
    batch_size = args.batch_size
    num_iterations = args.niter
    c_puct = args.c_puct
    jumpout = args.jumpout
    pdb_id = args.pdbid
    dual_pdbid = args.dual_pdbid
    start_sequence = args.start_sequence

    for init_num in init_num_list:
        init_num = int(init_num)
        seed = 50 + init_num
        np.random.seed(seed)
        random.seed(seed)

        peptide_length = args.length
        start_pool = []
        pool_size = 1

        if start_sequence is None:
            for _ in range(pool_size):
                seq_weights, peptide_sequence = initialize_weights(peptide_length)
                start_sequence = peptide_sequence
                start_pool.append(start_sequence)
        else:
            start_pool.append(start_sequence)

        start_sequence = start_pool[0]
        output_directory = f"{base_output_directory}"
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        with open(f'{output_directory}/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        training_pipeline = TrainingPipeline(pdb_id, dual_pdbid, start_sequence, num_playouts, batch_size, AMINO_ACIDS, num_iterations, c_puct, jumpout)
        training_pipeline.run(output_directory=output_directory)

        end_time = datetime.now()
        print(f"Output directory: \n{output_directory}")
        print("Total time taken: {}s".format(end_time - start_time))
        print("The current date and time is", datetime.now())