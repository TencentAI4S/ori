# -*- coding: utf-8 -*-
from typing import List, Union
import numpy as np
import copy
import time

AAS = "ARNDCQEGHILKMFPSTWYV"

def one_hot_to_string(one_hot: Union[List[List[int]], np.ndarray], alphabet: str) -> str:
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state, mp_dict):
        node = self._root
        playout_seqs = []
        playout_loss = []
        state.playout_dict.update(mp_dict)
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            state.do_mutate(action, playout=1)
            playout_seq = one_hot_to_string(state._state, AAS)
            playout_seqs.append(playout_seq)
            playout_loss.append(state.loss)

        action_probs, leaf_value = self._policy(state)
        end = state.mutation_end()
        if not end:
            node.expand(action_probs)
        else:

            leaf_value = state._state_fitness

        node.update_recursive(-leaf_value)
        re_m_p_dict = copy.deepcopy(state.playout_dict)
        return playout_seqs, playout_loss, re_m_p_dict

    def get_move_probs(self, state, m_p_dict, temp=1e-3):

        play_seq_list = []
        play_loss_list = []
        g_m_p_dict = m_p_dict
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            state_copy.playout = 1
            t_0 = time.time()
            play_seq, play_loss, mp_dict = self._playout(state_copy, g_m_p_dict)
            play_seq_list.extend(play_seq)
            play_loss_list.extend(play_loss)
            g_m_p_dict.update(mp_dict)


        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        if not act_visits:
            return [], []
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs, play_seq_list, play_loss_list, g_m_p_dict

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

class MCTSMutater(object):

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.m_p_dict = {}

    def set_player_ind(self, p):
        self.player = p

    def reset_Mutater(self):
        self.mcts.update_with_move(-1)

    def get_action(self, Seq_env, temp=1e-3, return_prob=0):
        move_probs = np.zeros(Seq_env.seq_len*Seq_env.vocab_size)

        get_move_mp_dict = copy.deepcopy(self.m_p_dict)
        acts, probs, play_seqs, play_losses, m_p_dict = self.mcts.get_move_probs(Seq_env, get_move_mp_dict, temp) #
        self.m_p_dict.update(m_p_dict)

        if acts:
            move_probs[list(acts)] = probs
            if self._is_selfplay:

                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
                # print("AI move: %d\n" % (move))
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)


            if return_prob:
                return move, move_probs, play_seqs, play_losses
            else:
                return move, play_seqs, play_losses
        else:
            return [],[], play_seqs, play_losses

    def __str__(self):
        return "MCTS {}".format(self.player)