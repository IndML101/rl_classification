import numpy as np
import gym


class TabularEnv(gym.Env):
    def __init__ (self, dataset):
        super(TabularEnv, self).__init__()

        self.x_train = dataset[0].values
        self.y_train = dataset[1].values

        self.action_space = gym.spaces.Discrete(self.x_train.shape[1]*2, )
        self.observation_shape = (self.x_train.shape[1] * 3, ) 
        self.observation_space = gym.spaces.Box(low= -np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float64)
        self.num_actions = self.x_train.shape[1]*2
        self.train_episodes = np.concatenate((self.x_train, np.zeros(self.x_train.shape), self.x_train), axis=1)
        self.m = self.x_train.shape[1]
        self.step_count = 0
        self.dataset_idx = -1
        # self.flag = bool(self.y_train.iloc[self.dataset_idx].all())

    def calc_reward(self):

        # self.flag = bool(self.y_train.iloc[self.dataset_idx].all())
        # flag = 1

        self.z = sum(self.train_episodes[self.dataset_idx,:] == 0)

        # if self.flag:
        #     return 1-(self.z/self.m)Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
        # else:
        #     return (self.z/self.m)-1

        return 1-(self.z/self.m)

    def step(self, action):

        done = False

        if action < self.x_train.shape[1]:
            self.train_episodes[self.dataset_idx, self.x_train.shape[1]+action] = self.train_episodes[self.dataset_idx, action]
        else:
            self.train_episodes[self.dataset_idx, self.x_train.shape[1]+action] = 0

        self.reward = self.calc_reward()

        self.step_count += 1
        if self.step_count == self.x_train.shape[1]:
            done = True
            # self.flag = bool(self.y_train.iloc[self.dataset_idx].all())

        return self.train_episodes[self.dataset_idx,:], self.reward, done, {}

    def reset(self):
        # print("reset"”)
        
        self.step_count = 0
        
        if self.dataset_idx < self.x_train.shape[0]-1:
            self.dataset_idx += 1
        else:
            self.dataset_idx = 0
        try:
            self.flag = self.y_train[self.dataset_idx]
        except:
            self.flag = self.y_train[self.dataset_idx].all()

        return self.train_episodes[self.dataset_idx, :]

    def close(self):
        self.step_count = 0
        self.dataset_idx = -1
        self.train_episodes = np.concatenate((self.x_train, np.zeros(self.x_train.shape), self.x_train), axis=1)
        
        return


