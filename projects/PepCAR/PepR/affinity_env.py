# -*- coding: utf-8 -*-

import numpy as np
import torch
import random
import json
from typing import List, Union
from iupred.calculate_matrix_single import fasta_2_matrix
from esm.extract_pep_esm import get_peptide_embedding
from esm.extract_pro_esm import get_protein_embedding
from pep_to_smiles import pep_to_smile
from graph.pep_graph import featurize_peptide_graph
from graph.smi_graph import featurize_drug_graph
from graph.pro_graph import featurize_protein_graph
from PepC.net import ASMNet
from PepC.amnet.net import AMNet
from PepC.smnet.net import SMNet
import residue_constants
import sys
import copy
from transformers import logging
import os

logging.set_verbosity_warning()
logging.set_verbosity_error()
AAS = "ARNDCQEGHILKMFPSTWYV"

def mutate_sequence(peptide_sequence, ex_list):
    restypes = np.array(list(AAS))
    seqlen = len(peptide_sequence)

    while True:
        seeds = peptide_sequence
        pi_s = np.random.choice(np.arange(seqlen), 3, replace=False)
        for pi in pi_s:
            aa = np.random.choice(restypes, replace=False)
            seeds = seeds[:pi] + aa + seeds[pi + 1:]
        if seeds not in ex_list:
            break
    return seeds

def update_features(peptide_sequence):
    ids = fasta_2_matrix(peptide_sequence)
    smiles = pep_to_smile(peptide_sequence)
    emb = get_peptide_embedding(peptide_sequence)
    pep_data = featurize_peptide_graph(peptide_sequence, ids, emb)
    smi_data = featurize_drug_graph(smiles)
    return pep_data, smi_data

def predict_function(peptide_sequence, prot_data, models):
    pep_data, smi_data = update_features(peptide_sequence)
    y_mean = 0
    with torch.no_grad():
        for model in models:
            model.eval()
            y = model(prot_data, pep_data, smi_data)
            y_mean += y.cpu().item()
    return y_mean / len(models)

def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out

def one_hot_to_string(one_hot: Union[List[List[int]], np.ndarray], alphabet: str) -> str:
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])

class Seq_env(object):
    """sequence space for the env"""
    def __init__(self, seq_len, alphabet, pdbid, starting_seq):
        self.move_count = 0
        self.seq_len = seq_len
        self.vocab_size = len(alphabet)
        self.alphabet = alphabet
        self.starting_seq = starting_seq
        self.seq = starting_seq

        aa = residue_constants.sequence_to_onehot(
            sequence=self.seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True
        )
        self.init_state = aa[:, 0:-1]
        self.previous_init_state = aa[:, 0:-1]
        self.init_state_count = 0

        self.unuseful_move = 0
        self.repeated_seq_ocurr = False
        self.episode_seqs = [starting_seq]
        self.loss = 0.0
        self.playout_dict = {}
        self.move_0_flag = False

        self.random_seed = random.randrange(sys.maxsize)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.load_models(device)

        self.prot_data = self.load_protein_data(pdbid)

    def load_models(self, device):
        amnets = [AMNet() for _ in range(5)]
        smnets = [SMNet() for _ in range(5)]
        PepCs = [ASMNet(amnets[i], smnets[i]) for i in range(5)]

        pretrained_path = '../model_weights/PepC'
        for i in range(5):
            PepCs[i].load_state_dict(torch.load(f'{pretrained_path}/fold_{i + 1}.pt', map_location=device), strict=False)

        print(f"################## model loaded on {device} #####################")
        return PepCs

    def load_protein_data(self, pdb_id):
        esm_path = '../PepC/receptor_data/esm'
        coord_json = '../PepC/receptor_data/coordinates.json'
        epitope_path = '../PepC/receptor_data/rec_interface.json'
        rec_path = '../PepC/receptor_data/mod_rec_seq.json'

        coord_data = json.load(open(coord_json, 'r'))
        epitope_data = json.load(open(epitope_path, 'r'))
        rec_data = json.load(open(rec_path, 'r'))
        esm_embed_path = f"{esm_path}/{pdb_id}.pt"
        if not os.path.exists(esm_embed_path):
            esm_embed = get_peptide_embedding(rec_data[pdb_id])
        else:
            esm_embed = torch.load(esm_embed_path)

        _coords = coord_data[pdb_id]
        entry = {
            "seq": rec_data[pdb_id],
            "coords": list(zip(_coords["N"], _coords["CA"], _coords["C"], _coords["O"])),
            "embed": esm_embed,
            "epitope": epitope_data[pdb_id]
        }
        return featurize_protein_graph(entry)

    def init_seq_state(self):
        self.previous_fitness = -float("inf")
        self.move_count = 0
        self.unuseful_move = 0
        self.repeated_seq_ocurr = False
        self._state = copy.deepcopy(self.init_state)

        combo = one_hot_to_string(self._state, AAS)
        self.init_combo = combo
        self.episode_seqs.append(combo)

        y = predict_function(combo, self.prot_data, self.models)
        self.loss = 1 / y
        reward = y
        self._state_fitness = reward / 1000
        self.availables = list(range(self.seq_len * self.vocab_size))

        self.update_availables(combo)

        self.states = {}
        self.last_move = -1
        self.previous_init_state = copy.deepcopy(self._state)

    def update_availables(self, combo):
        for i, a in enumerate(combo):
            self.availables.remove(self.vocab_size * i + AAS.index(a))
        for e_s in self.episode_seqs:
            a_e_s = string_to_one_hot(e_s, AAS)
            a_e_s_ex = np.expand_dims(a_e_s, axis=0)
            if 'nda' not in locals():
                nda = a_e_s_ex
            else:
                nda = np.concatenate((nda, a_e_s_ex), axis=0)

        c_i_s = string_to_one_hot(combo, AAS)
        for i, aa in enumerate(combo):
            tmp_c_i_s = np.delete(c_i_s, i, axis=0)
            for slice in nda:
                tmp_slice = np.delete(slice, i, axis=0)
                if (tmp_c_i_s == tmp_slice).all():
                    bias = np.where(slice[i] != 0)[0][0]
                    to_be_removed = self.vocab_size * i + bias
                    if to_be_removed in self.availables:
                        self.availables.remove(to_be_removed)

    def current_state(self):
        return self._state.T

    def do_mutate(self, move, playout=0):
        self.previous_fitness = self._state_fitness
        self.move_count += 1
        self.availables.remove(move)
        pos = move // self.vocab_size
        res = move % self.vocab_size

        if self._state[pos, res] == 1:
            self.unuseful_move = 1
            self._state_fitness = 0.0
        else:
            self._state[pos] = 0
            self._state[pos, res] = 1
            combo = one_hot_to_string(self._state, AAS)
            
            if playout==0:
                if combo not in self.playout_dict.keys():
                    y = predict_function(combo, self.prot_data, self.models)
                    self.loss = 1/y
                    reward = y
                    self._state_fitness = reward/1000
                else:
                    self._state_fitness = self.playout_dict[combo][0]
                    self.loss = 1/(1000*self._state_fitness)
            else:
                if combo not in self.playout_dict.keys():

                    y = predict_function(combo, self.prot_data, self.models)
                    self.loss = 1/y
                    reward = y

                    self._state_fitness = reward/1000
                    self.playout_dict[combo] = [reward/1000]
                else:
                    self._state_fitness = self.playout_dict[combo][0]
                    self.loss = 1/(1000*self._state_fitness)

        current_seq = one_hot_to_string(self._state, AAS)
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            self._state_fitness = 0.0
        else:
            self.episode_seqs.append(current_seq)
        if self._state_fitness > self.previous_fitness:
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0
        if move==0:
            self.move_0_flag = True
        self.last_move = move

    def mutation_end(self):
        return self.repeated_seq_ocurr or self.unuseful_move == 1 or self._state_fitness < self.previous_fitness or self.move_0_flag


class Mutate(object):
    """mutating server"""

    def __init__(self, Seq_env, **kwargs):
        self.Seq_env = Seq_env

    def start_p_mutating(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.Seq_env.init_board(start_player)
        p1, p2 = self.Seq_env.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.Seq_env, player1.player, player2.player)
        while True:
            current_player = self.Seq_env.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.Seq_env)
            self.Seq_env.do_move(move)
            if is_shown:
                self.graphic(self.Seq_env, player1.player, player2.player)
            end, winner = self.Seq_env.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_mutating(self, mutater, is_shown=0, temp=1e-3, jumpout=50):
        """ start mutating using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """

        if (self.Seq_env.previous_init_state == self.Seq_env.init_state).all():
            self.Seq_env.init_state_count += 1
        if self.Seq_env.init_state_count >= jumpout:
            print("Random start replacement****")
            current_start_seq = one_hot_to_string(self.Seq_env.init_state, AAS)
            episode_seqs = copy.deepcopy(self.Seq_env.episode_seqs)
            playout_seqs = copy.deepcopy(list(self.Seq_env.playout_dict.keys()))
            e_p_list = list(set(episode_seqs + playout_seqs))
            new_start_seq = mutate_sequence(current_start_seq, e_p_list)
            self.Seq_env.init_state = string_to_one_hot(new_start_seq, self.Seq_env.alphabet).astype(np.float32)
            self.Seq_env.init_state_count = 0

        self.Seq_env.init_seq_state()
        # print("Start sequence：{}".format(self.Seq_env.init_combo))
        generated_peptide = []

        fit_result = []
        play_seqs_list = []
        play_losses_list = []

        states, mcts_probs, reward_z = [], [], [] #, current_players #, []

        while True:
            move, move_probs, play_seqs, play_losses = mutater.get_action(self.Seq_env,   #,m_p_dict
                                                 temp=temp,
                                                 return_prob=1)
            self.Seq_env.playout_dict.update(mutater.m_p_dict)
            if play_seqs:
                play_seqs_list.extend(play_seqs)
                play_losses_list.extend(play_losses)
            if move:
                states.append(self.Seq_env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(self.Seq_env._state_fitness)

                self.Seq_env.do_mutate(move)
                generated_peptide.append(one_hot_to_string(self.Seq_env._state, AAS))

                fit_result.append(self.Seq_env._state_fitness)

                state_string = one_hot_to_string(self.Seq_env._state, AAS)
                print("Mutated seq", state_string)


            end = self.Seq_env.mutation_end()
            if end:
                mutater.reset_Mutater()
                if is_shown:

                    print("Mutation end.")
                playout_dict = copy.deepcopy(self.Seq_env.playout_dict)
                return zip(states, mcts_probs, reward_z), zip(generated_peptide, fit_result), play_seqs_list, play_losses_list, playout_dict