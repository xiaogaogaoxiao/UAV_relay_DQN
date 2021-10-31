import gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pylab
import matplotlib.pyplot as plt
USE_GPU = True

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)


TARGET_REPLACE_ITER = 50  # target update frequency 100


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        self.linear1.weight.data.normal_(0, 0.1)
        self.linear2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.eval_net = Net(self.state_space_dim, 256, self.action_space_dim)
        self.target_net = Net(self.state_space_dim, 256, self.action_space_dim)#.to(device)  # 256
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.buffer = []
        self.learn_step_counter = 0
        self.steps = 0

    def act(self, s0):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-2.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(s0, dtype=torch.float,device=device).view(1, -1)
            a0 = torch.argmax(self.eval_net(s0)).item()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if (len(self.buffer)) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)
        s0, a0, r1, s1 = zip(*samples)
        s0 = torch.tensor(s0, dtype=torch.float,device=device)
        a0 = torch.tensor(a0, dtype=torch.long).view(self.batch_size, -1)
        r1 = torch.tensor(r1, dtype=torch.float,device=device).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float,device=device)

        #===================================================================#
        # y_next_Q = self.eval_net(s1)
        # argmax_Q = np.argmax(y_next_Q.data.numpy(), axis=1)
        # q_next = self.target_net(s1).detach()
        #
        # q_next_numpy = q_next.data.numpy()
        # q_updata = np.zeros((self.batch_size, 1))
        # for i in range(self.batch_size):
        #     q_updata[i] = q_next_numpy[i, argmax_Q[i]]
        #
        # q_updata = self.gamma * q_updata
        # q_updata = torch.tensor(q_updata, dtype=torch.float, device=device)
        # y_true = r1 + q_updata
        # ========================================================#
        y_true = r1 + self.gamma * torch.max(self.target_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        y_pred = self.eval_net(s0).gather(1, a0)

        loss_fn = nn.MSELoss()
        loss = loss_fn(y_pred, y_true)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target net update
        self.learn_step_counter += 1

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:

            self.target_net.load_state_dict(self.eval_net.state_dict())


    def net_reset(self):
        self.eval_net.reset_parameters()
        self.target_net.reset_parameters()
        self.buffer = []
        self.learn_step_counter = 0
        self.steps = 0

    def save_model(self, filename, directory):

        torch.save(self.eval_net.state_dict(), '%s/%s_eval_net.pth' % (directory, filename))

    def load_model(self, filename, directory):

        self.eval_net.load_state_dict(torch.load('%s/%s_eval_net.pth' % (directory, filename)))

    def plot(self, str, mean):

        pylab.figure(1)
        pylab.xlabel('Time slot')
        pylab.ylabel(str)
        pylab.plot(mean)
        pylab.title('')
        pylab.grid(True, linestyle='-.')
        pylab.legend(loc='best')

        pylab.show()

    def plot_compare(self, str, mean_0,mean_1,mean_2):

        pylab.figure(0)

        pylab.plot(mean_0,label='DQN')
        pylab.legend(loc='best')

        pylab.plot(mean_1, label='ATPC')
        pylab.legend(loc='best')
        pylab.plot(mean_2, label='P=0.7')
        pylab.legend(loc='best')
        pylab.title('')
        pylab.xlabel('Time slot')
        pylab.ylabel(str)
        pylab.grid(True,linestyle='-.')


        pylab.show()


score = []
mean = []
def main():
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    params = {
        'gamma': 0.8,
        'epsi_high': 0.8,
        'epsi_low': 0.05,
        'decay': 200,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    agent = Agent(**params)
    print('\nCollecting experience...')
    total_steps=0
    print(env.observation_space.shape[0],env.action_space.n)
    for episode in range(100):
        s0 = env.reset()
        print(s0)
        print('episode: %d total_steps: %d' % (episode, total_steps))
        total_reward = 1
        while True:
            env.render()
            total_steps += 1
            a0 = agent.act(s0)
            print(a0)
            s1, r1, done, _ = env.step(a0)
            x, x_dot, theta, theta_dot = s1
            r0 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r1 = r0 + r2
            if done:
                r1 = -1

            agent.put(s0, a0, r1, s1)

            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn()

        score.append(total_reward)
        print(total_reward)
        mean.append(sum(score[-5:]) /5)
    agent.plot(score, mean)

if __name__ == '__main__':
        main()