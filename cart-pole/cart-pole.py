import os
import sys
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cwd = os.path.dirname(os.path.abspath(__file__))
path_list = cwd.split('/')
project_folder_path = '/'.join(path_list[:-2])
sys.path.append(project_folder_path)


class Actor(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=32):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear3 = nn.Linear(hidden_size * 2, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        preference = F.relu(self.linear3(output))
        return preference


class Critic(nn.Module):

    def __init__(self, state_size, action_size, hidden_size=32):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linearl = nn.Linear(self.state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size * 2)
        self.linear3 = nn.Linear(hidden_size * 2, 1)

    def forward(self, state):
        output = F.relu(self.linearl(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


class M2ACLA:

    def __init__(self,
                 environment,
                 actor=None,
                 critic=None,
                 alpha=0.005,
                 beta=0.005,
                 gamma=0.9,
                 tau=0.05):
        self.env = environment
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        if actor is None:
            self.actor = Actor(state_size=self.state_size,
                               action_size=self.action_size)
        else:
            self.actor = actor

        if critic is None:
            self.critic = Critic(state_size=self.state_size,
                                 action_size=self.action_size)
        else:
            self.critic = critic

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def act(self, preference):

        # if self.env.flag:
        #     return Categorical(F.softmax(torch.div(preference,self.tau), dim=-1)).sample()
        # else:
        #     return Categorical(F.softmax(torch.div(torch. negative(preference),self.tau), dim=-1)).sample()

        return Categorical(F.softmax(torch.div(preference, self.tau),
                                     dim=-1)).sample()

    def compute_v_target(self, next_value, reward, done):
        if done:
            return reward
        else:
            return reward + next_value * self.gamma

    def compute_q_target(self, value, action):
        # value = value.detach().numpy()[@]
        eye = torch.eye(self.action_size)
        if value >= 0.0:
            # return torch.FloatTensor([1.0])
            return eye[action, :].to(self.device)
        else:
            # return torch.FloatTensor([6.0])
            return torch.zeros(self.action_size).to(self.device)

    def init_weights(self, model):
        for name, param in model.named_parameters():
            if name == 'weight':
                nn.init.xavier_uniform_(param.data, 1.0)
            else:
                nn.init.ones_(param.data)

    def learn(self, timesteps=1000, model_path=None):
        self.actor.apply(self.init_weights)
        self.critic.apply(self.init_weights)
        self.actor = self.actor.to(device=self.device, dtype=torch.float64)
        self.critic = self.critic.to(device=self.device, dtype=torch.float64)

        optimizerA = optim.Adam(self.actor.parameters(), lr=self.alpha)
        optimizerC = optim.Adam(self.critic.parameters(), lr=self.beta)

        reward_list = list()
        for iter in tqdm.trange(timesteps):
            state = self.env.reset()
            done = False
            episode_rewards = list()
            # steps = 0
            while not done:
                state = torch.from_numpy(state).to(device=self.device,
                                                   dtype=torch.float64)
                preference, value = self.actor(state), self.critic(state)

                action = self.act(preference)

                next_state, reward, done, _ = self.env.step(
                    action.cpu().detach().numpy())
                next_state = torch.from_numpy(next_state).to(
                    device=self.device, dtype=torch.float64)
                next_value = self.critic(next_state)

                critic_loss = self.compute_v_target(next_value, reward,
                                                    done) - value
                actor_loss = (self.compute_q_target(critic_loss, action) -
                              preference).sum()

                optimizerA.zero_grad()
                optimizerC.zero_grad()

                actor_loss.backward()
                critic_loss.backward()

                optimizerA.step()
                optimizerC.step()

                state = next_state.cpu().detach().numpy()
                episode_rewards.append(reward)
                # done = False
                # steps += 1

            reward_list.append(np.sum(episode_rewards))

        if model_path is not None:
            torch.save(self.actor, os.path.join(model_path, 'actor.pkl'))
            torch.save(self.critic, os.path.join(model_path, 'critic.pkl'))

        self.env.close()

        return self.actor, self.critic, reward_list

    def evaluate_value(self, environment):

        value_0 = list()

        for i in tqdm.trange(1):
            state = environment.reset()
            state = torch.from_numpy(state).to(self.device)
            value = self.critic(state)
            value_0.append(value.cpu().detach().numpy())

        environment.close()

        return np.array(value_0)

    def evaluate_reward(self, environment):

        all_episode_rewards = list()
        rewards = list()

        for j in tqdm.trange(1):
            episode_rewards = list()
            done = False
            state = environment.reset()
            state = torch.from_numpy(state).to(self.device)
            # steps = 0
            while not done:
                preference = self.actor(state)
                action = self.act(preference)
                state, reward, done, info = environment.step(action)
                environment.render()
                episode_rewards.append(reward)
                # done =False
                # steps += 1

            all_episode_rewards.append(episode_rewards)
            rewards.append(reward)
            state = environment.reset()

        mean_episode_reward = np.sum(rewards)
        print("Mean reward:", mean_episode_reward)
        environment.close()

        return rewards, all_episode_rewards


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    m2acla = M2ACLA(env)
    timesteps = 1000
    actor, critic, rewards = m2acla.learn(
        timesteps=timesteps,
        model_path=os.path.join(project_folder_path,
                                'rl_classification/models/cart-pole/'))

    fig, ax = plt.subplots(figsize=[10, 7])
    plt.plot(list(range(timesteps)), rewards, marker='.')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig(
        os.path.join(project_folder_path,
                     'rl_classification/cart-pole/rewards-plot.png'))
