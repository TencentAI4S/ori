# -*- coding: utf-8 -*-
import copy
from typing import List, Union

import numpy as np

AAS = "ARNDCQEGHILKMFPSTWYV"


def one_hot_to_string(one_hot: Union[List[List[int]], np.ndarray], alphabet: str) -> str:
    indices = np.argmax(one_hot, axis=1)
    return "".join(alphabet[i] for i in indices)


def _normalized_exp(scores):
    shifted = scores - np.max(scores)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


class Node:
    """Single node of the search tree holding visit statistics and a prior."""

    def __init__(self, parent, prior):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.q_value = 0.0
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, priors):
        for action, prob in priors:
            self.children.setdefault(action, Node(self, prob))

    def score(self, c_puct):
        exploration = c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        return self.q_value + exploration

    def best_child(self, c_puct):
        return max(self.children.items(), key=lambda item: item[1].score(c_puct))

    def backup(self, value):
        """Propagate an evaluation to this node and its ancestors, flipping sign
        at every level to account for the alternating perspective."""
        node, signed = self, value
        while node is not None:
            node.visits += 1
            node.q_value += (signed - node.q_value) / node.visits
            node, signed = node.parent, -signed


class MCTS:
    """Monte Carlo Tree Search driven by a policy-value estimator."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self.root = Node(None, 1.0)
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def _simulate(self, state, shared_dict):
        visited_seqs = []
        visited_losses = []
        state.playout_dict.update(shared_dict)

        node = self.root
        while not node.is_leaf():
            action, node = node.best_child(self.c_puct)
            state.do_mutate(action, playout=1)
            visited_seqs.append(one_hot_to_string(state._state, AAS))
            visited_losses.append(state.loss)

        priors, value = self.policy_value_fn(state)
        if state.mutation_end():
            value = state._state_fitness
        else:
            node.expand(priors)

        node.backup(-value)
        return visited_seqs, visited_losses, copy.deepcopy(state.playout_dict)

    def search(self, state, shared_dict, temp=1e-3):
        seq_log = []
        loss_log = []
        rolling_dict = shared_dict
        for _ in range(self.n_playout):
            rollout_state = copy.deepcopy(state)
            rollout_state.playout = 1
            seqs, losses, updated = self._simulate(rollout_state, rolling_dict)
            seq_log.extend(seqs)
            loss_log.extend(losses)
            rolling_dict.update(updated)

        visit_stats = [(action, child.visits) for action, child in self.root.children.items()]
        if not visit_stats:
            return [], []
        actions, visits = zip(*visit_stats)
        probs = _normalized_exp((1.0 / temp) * np.log(np.array(visits) + 1e-10))
        return actions, probs, seq_log, loss_log, rolling_dict

    def advance(self, move):
        child = self.root.children.get(move)
        if child is not None:
            child.parent = None
            self.root = child
        else:
            self.root = Node(None, 1.0)


class MCTSMutater:
    """Selects mutations for the sequence environment using MCTS rollouts."""

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.m_p_dict = {}

    def set_player_ind(self, p):
        self.player = p

    def reset_Mutater(self):
        self.mcts.advance(-1)

    def _pick_move(self, actions, probs):
        if self._is_selfplay:
            noise = np.random.dirichlet(0.3 * np.ones(len(probs)))
            move = np.random.choice(actions, p=0.75 * probs + 0.25 * noise)
            self.mcts.advance(move)
        else:
            move = np.random.choice(actions, p=probs)
            self.mcts.advance(-1)
        return move

    def get_action(self, Seq_env, temp=1e-3, return_prob=0):
        move_probs = np.zeros(Seq_env.seq_len * Seq_env.vocab_size)

        result = self.mcts.search(Seq_env, copy.deepcopy(self.m_p_dict), temp)
        actions, probs, play_seqs, play_losses, shared_dict = result
        self.m_p_dict.update(shared_dict)

        if not actions:
            return [], [], play_seqs, play_losses

        move_probs[list(actions)] = probs
        move = self._pick_move(actions, probs)

        if return_prob:
            return move, move_probs, play_seqs, play_losses
        return move, play_seqs, play_losses

    def __str__(self):
        return "MCTS {}".format(self.player)
