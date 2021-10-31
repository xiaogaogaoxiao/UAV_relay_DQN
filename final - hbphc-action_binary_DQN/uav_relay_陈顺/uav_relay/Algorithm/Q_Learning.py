import random

import numpy as np
import copy


class q_learning:
    def __init__(self, n_state, n_action, alpha=0.7, alpha_q = 0.3, epsilon=0.1, gamma=0.9, deta = 0.02, value_init=0,value_n_actions = 0.5):
        self.n_states = n_state
        self.n_actions = n_action
        self.alpha = alpha
        self.alpha_q = alpha_q
        self.epsilon = epsilon
        self.gamma = gamma
        self.deta = deta
        self.value_init = value_init
        self.value_n_actions = value_n_actions
        self.q_table = np.ones((self.n_states, self.n_actions)) * self.value_init
        self.pi_table = np.ones((self.n_states, self.n_actions)) * self.value_n_actions

    def action_choose(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() > epsilon:
            max_actions = np.where(self.q_table[state] == self.q_table[state].max())[0]
            action = np.random.choice(max_actions)

        else:
            action = np.random.randint(self.n_actions)
        return action

    def action_choose_phc(self, state, epsilon = None):
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() > epsilon:
            action = np.random.choice(np.arange(2), p=self.pi_table[state])
        else:
            action = np.random.randint(self.n_actions)
        # action = np.random.choice(max_actions)
        return action


    def table_learn(self, state, action, state_next, reward):
        q_eval = self.q_table[state, action]
        q_target = reward + self.gamma * self.q_table[state_next].max()
        self.q_table[state, action] = (1-self.alpha_q) * q_eval + self.alpha_q * q_target





    def table_learn_phc(self, state, action, state_next, reward):
        q_eval = self.q_table[state, action]
        q_target = reward + self.gamma * self.q_table[state_next].max()
        self.q_table[state, action] = (1-self.alpha) * q_eval + self.alpha * q_target
        max_actions = np.where(self.q_table[state] == self.q_table[state].max())[0]
        action_new = np.random.choice(max_actions)
        self.pi_table[state, action_new] = self.pi_table[state, action_new] + self.deta
        if self.pi_table[state, action_new] > 1:
            self.pi_table[state] = 0.1
            self.pi_table[state, action_new] = 0.9
        else:
            self.pi_table[state] = self.pi_table[state] - self.deta / (self.n_actions - 1)
            self.pi_table[state, action_new] = self.pi_table[state, action_new] + self.deta / (self.n_actions - 1)
            for i in range(self.n_actions):
                if self.pi_table[state,i] < 0:
                    self.pi_table[state,action_new] = 1
                    self.pi_table[state,i] = 0





    def reset(self):
        self.q_table = np.ones((self.n_states, self.n_actions)) * self.value_init
        self.pi_table = np.ones((self.n_states, self.n_actions)) * self.value_n_actions

    def load(self,Q_table,pi_table):
        self.q_table = Q_table.copy()
        self.pi_table = pi_table.copy()


    def save_q(self):
        return self.q_table

    def save_pi(self):
        return self.pi_table


    def load_q(self, Q_table):
        self.q_table = Q_table

    def load_pi(self,PI_table):
        self.pi_table = PI_table