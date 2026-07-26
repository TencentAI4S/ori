from __future__ import print_function

import argparse
import json
import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
from transformers import logging

from affinity_env import Seq_env, Mutate
from mcts_mutate import MCTSMutater
from pvnet import PolicyValueNet

logging.set_verbosity_warning()
logging.set_verbosity_error()

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"


def parse_fixed_positions(spec, seq_len):
    """Turn a 1-based position spec such as '1,5,13-17' into 0-based indices."""
    if not spec:
        return set()

    positions = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            first, last = (bound.strip() for bound in part.split('-', 1))
        else:
            first = last = part
        try:
            low, high = int(first), int(last)
        except ValueError:
            raise ValueError(f"invalid position spec '{part}'; expected e.g. '5' or '13-17'")
        if low > high:
            raise ValueError(f"invalid position range '{part}'; start is after end")
        if low < 1 or high > seq_len:
            raise ValueError(
                f"position range '{part}' is outside 1-{seq_len} for the given peptide"
            )
        positions.update(range(low - 1, high))
    return positions


def initialize_weights(peptide_length):
    """Sample a random starting peptide from Gumbel-perturbed logits."""
    logits = np.random.gumbel(0, 1, (peptide_length, 20))
    weights = np.array([np.exp(row) / np.sum(np.exp(row)) for row in logits])
    amino_acids = np.array(list(AMINO_ACIDS))
    peptide_sequence = "".join(amino_acids[np.argmax(weights, axis=1)])
    return weights, peptide_sequence


class TrainingPipeline:
    def __init__(self, pdb_id, dual_pdbid, initial_sequence, num_playouts, batch_size,
                 alphabet, num_iterations, c_puct, jumpout, initial_model=None, lam=0.5,
                 fixed_positions=None):
        self.pdb_id = pdb_id
        self.dual_pdbid = dual_pdbid
        self.sequence_length = len(initial_sequence)
        self.vocabulary_size = len(alphabet)
        self.environment = Seq_env(
            self.sequence_length, alphabet, self.pdb_id, self.dual_pdbid, initial_sequence,
            lam=lam, fixed_positions=fixed_positions,
        )
        self.mutation_strategy = Mutate(self.environment)

        self.learning_rate = 2e-3
        self.lr_multiplier = 1.0
        self.num_playouts = num_playouts
        self.c_puct = c_puct
        self.jumpout = jumpout
        self.batch_size = batch_size
        self.data_buffer = deque(maxlen=num_iterations)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_target = 0.02
        self.game_batch_count = num_iterations

        self.buffer_no_extend = False
        self.episode_length = 0

        self.generated_peptides = []
        self.loss_history = []
        self.loss_A_list = []
        self.loss_B_list = []
        self.play_sequence_history = []
        self.play_loss_history = []
        self.policy_dict = {}
        self.mutation_policy_dict = {}

        if initial_model:
            self.policy_value_net = PolicyValueNet(
                self.sequence_length, self.vocabulary_size,
                model_file=initial_model, use_gpu=True,
            )
        else:
            self.policy_value_net = PolicyValueNet(
                self.sequence_length, self.vocabulary_size, use_gpu=True
            )
        self.mcts_player = MCTSMutater(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct, n_playout=self.num_playouts, is_selfplay=1,
        )

    def _record_peptides(self, peptides):
        for peptide, fitness, loss_a, loss_b in peptides:
            if peptide not in self.generated_peptides:
                self.generated_peptides.append(peptide)
                self.loss_history.append(fitness)
                self.loss_A_list.append(loss_a)
                self.loss_B_list.append(loss_b)
            if peptide not in self.mutation_policy_dict:
                self.mutation_policy_dict[peptide] = [fitness, loss_a, loss_b]

    def collect_selfplay_data(self, num_games=1):
        self.buffer_no_extend = False
        for _ in range(num_games):
            play_data, peptides, play_sequences, play_losses, policy_dict = \
                self.mutation_strategy.start_mutating(self.mcts_player)
            play_data = list(play_data)
            self.episode_length = len(play_data)
            self.policy_dict = policy_dict
            self.mutation_policy_dict.update(self.policy_dict)

            if self.episode_length == 0:
                self.buffer_no_extend = True
                continue

            self.data_buffer.extend(play_data)
            self._record_peptides(peptides)
            self.play_sequence_history.extend(play_sequences)
            self.play_loss_history.extend(play_losses)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [entry[0] for entry in mini_batch]
        mcts_probs_batch = [entry[1] for entry in mini_batch]
        winner_batch = [entry[2] for entry in mini_batch]

        old_probs, _ = self.policy_value_net.policy_value(state_batch)
        kl_divergence = 0.0
        loss = entropy = None
        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch,
                self.learning_rate * self.lr_multiplier,
            )
            new_probs, _ = self.policy_value_net.policy_value(state_batch)
            kl_divergence = np.mean(np.sum(
                old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                axis=1,
            ))
            if kl_divergence > self.kl_target * 4:
                break

        if kl_divergence > self.kl_target * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl_divergence < self.kl_target / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        return loss, entropy

    def _write_playout_table(self, output_directory):
        values = list(self.mutation_policy_dict.values())
        width = max(len(row) for row in values)
        padded = np.array([row + [0] * (width - len(row)) for row in values])
        fitness_column = 1 / (padded[:, 0] * 1000)

        raw = np.array(values)
        table = pd.DataFrame({
            "peptide": np.array(list(self.mutation_policy_dict.keys())),
            "loss": fitness_column,
            "losses_A": raw[:, 1],
            "losses_B": raw[:, 2],
        })
        table.to_csv(f"{output_directory}/Move_and_playout.csv", index=False)

    def run(self, output_directory="./output/init0/"):
        try:
            for iteration in range(self.game_batch_count):
                print("######### {}-th Play ###########".format(iteration + 1))
                self.collect_selfplay_data(self.play_batch_size)
                self._write_playout_table(output_directory)

                if len(self.data_buffer) > self.batch_size and not self.buffer_no_extend:
                    self.policy_update()
        except KeyboardInterrupt:
            print('\n\rTraining interrupted.')


def parse_args():
    parser = argparse.ArgumentParser(description='MCTS_Evobind')
    parser.add_argument("--output_dir", type=str,
                        default="/alldata/LChuang_data/workspace/Project/Protein_design/MCTS_evobind/output/",
                        help="Output data directory")
    parser.add_argument("--start_sequence", type=str, default=None, help="Starting peptide sequence")
    parser.add_argument("--pdbid", type=str, default='1ssc', help="PDB ID")
    parser.add_argument("--dual_pdbid", type=str, default='1ssc', help="DUAL PDB ID")
    parser.add_argument("--n_playout", type=int, default=11, help="Number of playouts")
    parser.add_argument("--init_num", type=str, default='1', help="Initialization seed")
    parser.add_argument("--batch_size", type=int, default=32, help="MCTS batch size")
    parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--jumpout", type=int, default=1000, help="Jumpout parameter (1000 for no jump)")
    parser.add_argument("--c_puct", type=float, default=3, help="Exploration parameter for MCTS")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Weight of the first target in the dual-target score: "
                             "lam * y(pdbid) + (1 - lam) * y(dual_pdbid)")
    parser.add_argument("--fixed_positions", type=str, default=None,
                        help="1-based residue positions that must not be mutated, e.g. "
                             "'13-17' or '1,5,13-17'. All positions are mutable by default")
    parser.add_argument("--sleep", type=int, default=0, help="Sleep time to avoid resource exhaustion")
    parser.add_argument("--length", type=int, help="Length of the peptide sequence")
    return parser.parse_args()


def main():
    args = parse_args()

    if int(args.sleep) > 0:
        time.sleep(args.sleep * 5)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    start_time = datetime.now()
    print("The current date and time is", start_time)

    start_sequence = args.start_sequence
    for init_num in args.init_num.split(','):
        seed = 50 + int(init_num)
        np.random.seed(seed)
        random.seed(seed)

        if start_sequence is None:
            _, start_sequence = initialize_weights(args.length)

        output_directory = f"{args.output_dir}"
        os.makedirs(output_directory, exist_ok=True)
        with open(f'{output_directory}/args.txt', 'w') as handle:
            json.dump(args.__dict__, handle, indent=2)

        fixed_positions = parse_fixed_positions(args.fixed_positions, len(start_sequence))

        pipeline = TrainingPipeline(
            args.pdbid, args.dual_pdbid, start_sequence, args.n_playout,
            args.batch_size, AMINO_ACIDS, args.niter, args.c_puct, args.jumpout,
            lam=args.lam, fixed_positions=fixed_positions,
        )
        pipeline.run(output_directory=output_directory)

        end_time = datetime.now()
        print(f"Output directory: \n{output_directory}")
        print("Total time taken: {}s".format(end_time - start_time))
        print("The current date and time is", datetime.now())


if __name__ == '__main__':
    main()
