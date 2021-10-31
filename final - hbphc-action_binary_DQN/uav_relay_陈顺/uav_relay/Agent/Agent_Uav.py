import sys

from Algorithm.Q_Learning import q_learning
import math


class agent:
    def __init__(self, n_state, n_action, epsilon_start=0.8, epsilon_end=0.01, decay=200,capacity = 100000):
        self.ql = q_learning(n_state, n_action)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay = decay
        self.step_count = 0
        self.buffer = []
        self.capacity = capacity

    def epsilon_anneal(self):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1.0 * self.step_count / self.decay)

    def choose_action_phc(self, state_results, epsilon=None):
        state_idx = state_results[0] + state_results[1] * 2
        if epsilon is None:
            epsilon = self.epsilon
            self.epsilon_anneal()
        action = self.ql.action_choose_phc(state_idx, epsilon)
        return action

    def choose_action(self, state_results, epsilon=None):
        state_idx = state_results[0] + state_results[1] * 2
        if epsilon is None:
            epsilon = self.epsilon
            self.epsilon_anneal()
        action = self.ql.action_choose(state_idx, epsilon)
        return action



    def learn_renew_phc(self, state_results, action, state_results_next, reward):
        self.step_count += 1
        state_idx = state_results[0] + state_results[1] * 2
        state_next_idx = state_results_next[0] + state_results_next[1] * 2
        self.ql.table_learn_phc(state_idx, action, state_next_idx, reward)

    def learn_renew(self, state_results, action, state_results_next, reward):
        self.step_count += 1
        state_idx = state_results[0] + state_results[1] * 2
        state_next_idx = state_results_next[0] + state_results_next[1] * 2
        self.ql.table_learn(state_idx, action, state_next_idx, reward)


    def reset(self):
        self.ql.reset()
        self.epsilon = self.epsilon_start
        self.step_count = 0

    def put_hb(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn_hb(self, state_results, action, state_results_next, reward):
        self.step_count += 1
        state_idx = state_results[0] + state_results[1] * 2
        state_next_idx = state_results_next[0] + state_results_next[1] * 2
        self.ql.table_learn(state_idx, action, state_next_idx, reward)


    def save_Q(self):
        print(self.ql.q_table.max())
        return self.ql.q_table

    def save_pi(self):
        print(self.ql.pi_table.max())
        return self.ql.pi_table


    def load(self, Q_table,pi_table):
        self.ql.load(Q_table,pi_table)
        self.epsilon = self.epsilon_start
        self.step_count = 0
