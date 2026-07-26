# -*- coding: utf-8 -*-

import copy
import json
import os
import random
import sys
from typing import List, Union

import numpy as np
import torch
from transformers import logging

import residue_constants
from esm.extract_pep_esm import get_peptide_embedding
from esm.extract_pro_esm import get_protein_embedding
from graph.pep_graph import featurize_peptide_graph
from graph.pro_graph import featurize_protein_graph
from graph.smi_graph import featurize_drug_graph
from iupred.calculate_matrix_single import fasta_2_matrix
from pep_to_smiles import pep_to_smile
from PepAF.amnet.net import AMNet
from PepAF.net import ASMNet
from PepAF.smnet.net import SMNet

logging.set_verbosity_warning()
logging.set_verbosity_error()

AAS = "ARNDCQEGHILKMFPSTWYV"

RECEPTOR_DIR = "../PepDAF/receptor_data"
PEPAF_WEIGHTS = "../model_weights/PepAF"
NUM_FOLDS = 5
DUAL_WEIGHT = 0.5


def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:
    matrix = np.zeros((len(sequence), len(alphabet)))
    for position, residue in enumerate(sequence):
        matrix[position, alphabet.index(residue)] = 1
    return matrix


def one_hot_to_string(one_hot: Union[List[List[int]], np.ndarray], alphabet: str) -> str:
    indices = np.argmax(one_hot, axis=1)
    return "".join(alphabet[i] for i in indices)


def mutate_sequence(peptide_sequence, ex_list, mutable_positions=None):
    """Randomly substitute three residues, resampling until the result is new.

    Substitutions are drawn only from `mutable_positions` when it is given, so a
    random restart cannot overwrite positions the user asked to keep.
    """
    residues = np.array(list(AAS))
    if mutable_positions is None:
        candidates = np.arange(len(peptide_sequence))
    else:
        candidates = np.array(sorted(mutable_positions), dtype=int)
    n_sites = min(3, len(candidates))
    if n_sites == 0:
        return peptide_sequence

    while True:
        candidate = peptide_sequence
        for position in np.random.choice(candidates, n_sites, replace=False):
            residue = np.random.choice(residues, replace=False)
            candidate = candidate[:position] + residue + candidate[position + 1:]
        if candidate not in ex_list:
            return candidate


def update_features(peptide_sequence):
    ids = fasta_2_matrix(peptide_sequence)
    smiles = pep_to_smile(peptide_sequence)
    emb = get_peptide_embedding(peptide_sequence)
    pep_data = featurize_peptide_graph(peptide_sequence, ids, emb)
    smi_data = featurize_drug_graph(smiles)
    return pep_data, smi_data


def predict_function(peptide_sequence, prot_data, dual_prot_data, models, lam=DUAL_WEIGHT):
    """Score a peptide against both receptors; `lam` weights the first target."""
    pep_data, smi_data = update_features(peptide_sequence)
    prot_sum = 0.0
    dual_sum = 0.0
    with torch.no_grad():
        for model in models:
            model.eval()
            prot_sum += model(prot_data, pep_data, smi_data).cpu().item()
            dual_sum += model(dual_prot_data, pep_data, smi_data).cpu().item()

    y_prot_mean = prot_sum / len(models)
    y_dual_mean = dual_sum / len(models)
    y_mean = lam * y_prot_mean + (1 - lam) * y_dual_mean
    return y_mean, y_prot_mean, y_dual_mean


class Seq_env(object):
    """Mutable peptide sequence acting as the RL environment state."""

    def __init__(self, seq_len, alphabet, pdbid, dual_pdbid, starting_seq, lam=DUAL_WEIGHT,
                 fixed_positions=None):
        self.move_count = 0
        self.seq_len = seq_len
        self.vocab_size = len(alphabet)
        self.alphabet = alphabet
        self.starting_seq = starting_seq
        self.seq = starting_seq
        self.lam = lam

        self.fixed_positions = set(fixed_positions or ())
        out_of_range = [p for p in self.fixed_positions if not 0 <= p < seq_len]
        if out_of_range:
            raise ValueError(
                f"fixed positions out of range for a {seq_len}-residue peptide: "
                f"{sorted(p + 1 for p in out_of_range)}"
            )
        self.mutable_positions = [p for p in range(seq_len) if p not in self.fixed_positions]
        if not self.mutable_positions:
            raise ValueError("every position is fixed; at least one position must stay mutable")

        onehot = residue_constants.sequence_to_onehot(
            sequence=self.seq,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )
        self.init_state = onehot[:, 0:-1]
        self.previous_init_state = onehot[:, 0:-1]
        self.init_state_count = 0

        self.unuseful_move = 0
        self.repeated_seq_ocurr = False
        self.episode_seqs = [starting_seq]
        self.loss = 0.0
        self.loss_A = 0.0
        self.loss_B = 0.0
        self.playout_dict = {}
        self.move_0_flag = False

        self.random_seed = random.randrange(sys.maxsize)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self.load_models(device)

        self.prot_data = self.load_protein_data(pdbid)
        self.dual_prot_data = self.load_protein_data(dual_pdbid)

    def load_models(self, device):
        ensemble = [
            ASMNet(AMNet(), SMNet())
            for _ in range(NUM_FOLDS)
        ]
        for fold, model in enumerate(ensemble, start=1):
            weights = torch.load(f"{PEPAF_WEIGHTS}/fold_{fold}.pt", map_location=device)
            model.load_state_dict(weights, strict=False)
        print(f"################## model loaded on {device} #####################")
        return ensemble

    def load_protein_data(self, pdb_id):
        coord_data = json.load(open(f"{RECEPTOR_DIR}/coordinates.json", "r"))
        epitope_data = json.load(open(f"{RECEPTOR_DIR}/rec_interface.json", "r"))
        rec_data = json.load(open(f"{RECEPTOR_DIR}/mod_rec_seq.json", "r"))

        embed_file = f"{RECEPTOR_DIR}/esm/{pdb_id}.pt"
        if os.path.exists(embed_file):
            esm_embed = torch.load(embed_file)
        else:
            esm_embed = get_peptide_embedding(rec_data[pdb_id])

        coords = coord_data[pdb_id]
        entry = {
            "seq": rec_data[pdb_id],
            "coords": list(zip(coords["N"], coords["CA"], coords["C"], coords["O"])),
            "embed": esm_embed,
            "epitope": epitope_data[pdb_id],
        }
        return featurize_protein_graph(entry)

    def _evaluate(self, combo, cache):
        """Return (fitness, loss, loss_A, loss_B) for a sequence, using the
        playout cache when available and optionally populating it."""
        if combo in self.playout_dict:
            fitness, loss_a, loss_b = self.playout_dict[combo]
            return fitness, 1 / (1000 * fitness), loss_a, loss_b

        y, y_prot, y_dual = predict_function(
            combo, self.prot_data, self.dual_prot_data, self.models, lam=self.lam
        )
        fitness = y / 1000
        result = (fitness, 1 / y, 1 / y_prot, 1 / y_dual)
        if cache:
            self.playout_dict[combo] = [fitness, 1 / y_prot, 1 / y_dual]
        return result

    def init_seq_state(self):
        self.previous_fitness = -float("inf")
        self.previous_loss_A = 1000000
        self.move_count = 0
        self.unuseful_move = 0
        self.repeated_seq_ocurr = False
        self._state = copy.deepcopy(self.init_state)

        combo = one_hot_to_string(self._state, AAS)
        self.init_combo = combo
        self.episode_seqs.append(combo)

        y, y_prot, y_dual = predict_function(
            combo, self.prot_data, self.dual_prot_data, self.models, lam=self.lam
        )
        self.loss = 1 / y
        self.loss_A = 1 / y_prot
        self.loss_B = 1 / y_dual
        self._state_fitness = y / 1000

        self.availables = list(range(self.seq_len * self.vocab_size))
        self.update_availables(combo)
        if self.fixed_positions:
            self.availables = [
                action for action in self.availables
                if action // self.vocab_size not in self.fixed_positions
            ]

        self.states = {}
        self.last_move = -1
        self.previous_init_state = copy.deepcopy(self._state)

    def update_availables(self, combo):
        for position, residue in enumerate(combo):
            self.availables.remove(self.vocab_size * position + AAS.index(residue))

        seen = np.stack([string_to_one_hot(seq, AAS) for seq in self.episode_seqs])
        current = string_to_one_hot(combo, AAS)
        for position in range(len(combo)):
            current_rest = np.delete(current, position, axis=0)
            for seq_onehot in seen:
                if (current_rest == np.delete(seq_onehot, position, axis=0)).all():
                    residue_idx = np.where(seq_onehot[position] != 0)[0][0]
                    action = self.vocab_size * position + residue_idx
                    if action in self.availables:
                        self.availables.remove(action)

    def current_state(self):
        return self._state.T

    def do_mutate(self, move, playout=0):
        self.previous_fitness = self._state_fitness
        self.previous_loss_A = self.loss_A
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
            fitness, loss, loss_a, loss_b = self._evaluate(combo, cache=bool(playout))
            self._state_fitness = fitness
            self.loss = loss
            self.loss_A = loss_a
            self.loss_B = loss_b

        current_seq = one_hot_to_string(self._state, AAS)
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            self._state_fitness = 0.0
            self.loss_A = 0.0
        else:
            self.episode_seqs.append(current_seq)

        if self._state_fitness > self.previous_fitness:
            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0
        if move == 0:
            self.move_0_flag = True
        self.last_move = move

    def mutation_end(self):
        return (
            self.repeated_seq_ocurr
            or self.unuseful_move == 1
            or self._state_fitness < self.previous_fitness
            or self.move_0_flag
        )


class Mutate(object):
    """Runs one self-play optimization episode over the sequence environment."""

    def __init__(self, Seq_env, **kwargs):
        self.Seq_env = Seq_env

    def _maybe_restart(self, jumpout):
        env = self.Seq_env
        if (env.previous_init_state == env.init_state).all():
            env.init_state_count += 1
        if env.init_state_count < jumpout:
            return
        print("Random start replacement****")
        current_start_seq = one_hot_to_string(env.init_state, AAS)
        excluded = list(set(
            copy.deepcopy(env.episode_seqs)
            + copy.deepcopy(list(env.playout_dict.keys()))
        ))
        new_start_seq = mutate_sequence(current_start_seq, excluded, env.mutable_positions)
        env.init_state = string_to_one_hot(new_start_seq, env.alphabet).astype(np.float32)
        env.init_state_count = 0

    def start_mutating(self, mutater, is_shown=0, temp=1e-3, jumpout=50):
        self._maybe_restart(jumpout)
        env = self.Seq_env
        env.init_seq_state()

        generated_peptide, fit_result = [], []
        play_seqs_list, play_losses_list = [], []
        loss_A, loss_B = [], []
        states, mcts_probs, reward_z = [], [], []

        while True:
            move, move_probs, play_seqs, play_losses = mutater.get_action(
                env, temp=temp, return_prob=1
            )
            env.playout_dict.update(mutater.m_p_dict)

            if play_seqs:
                play_seqs_list.extend(play_seqs)
                play_losses_list.extend(play_losses)

            if move:
                states.append(env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(env._state_fitness)
                loss_A.append(env.loss_A)
                loss_B.append(env.loss_B)

                env.do_mutate(move)
                generated_peptide.append(one_hot_to_string(env._state, AAS))
                fit_result.append(env._state_fitness)
                print("Mutated seq", one_hot_to_string(env._state, AAS))

            if env.mutation_end():
                mutater.reset_Mutater()
                if is_shown:
                    print("Mutation end.")
                playout_dict = copy.deepcopy(env.playout_dict)
                return (
                    zip(states, mcts_probs, reward_z),
                    zip(generated_peptide, fit_result, loss_A, loss_B),
                    play_seqs_list,
                    play_losses_list,
                    playout_dict,
                )
