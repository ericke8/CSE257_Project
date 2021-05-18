import numpy as np
import gym
import json
import os
import highway_env
import gym_maze

class Highway:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 1
                
        self.dims    = 25
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('highway-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (5, 5)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 1
        self.leaf_size    = 20
        self.kernel_type  = "rbf"
        self.gamma_type   = "scale"
        self.ninits       = 30
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        M = x.reshape(self.policy_shape)
        
        returns = []
        observations = []
        actions = []
        
        for i in range(self.num_rollouts):
            obs    = self.env.reset()
            done   = False
            totalr = 0.
            steps  = 0
            while not done:
                # M      = self.policy
                inputs = (obs - self.mean)/self.std
                
                action = np.argmax(np.sum(np.dot(M, inputs), axis=0))
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps  += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)
            
        return np.mean(returns)*-1
    
class Goddard:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 1
                
        self.dims    = 9
        self.lb      = np.zeros(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('gym_goddard:Goddard-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (3, 3)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 1
        self.leaf_size    = 20
        self.kernel_type  = "rbf"
        self.gamma_type   = "scale"
        self.ninits       = 30
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        M = x.reshape(self.policy_shape)
        
        returns = []
        observations = []
        actions = []
        
        for i in range(self.num_rollouts):
            obs    = self.env.reset()
            done   = False
            totalr = 0.
            steps  = 0
            while not done:
                # M      = self.policy
                inputs = (obs - self.mean)/self.std
                
                action = np.dot(M, inputs)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps  += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)
            
        return np.mean(returns)*-1
    

class Maze:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 1
                
        self.dims    = 8
        self.lb      = np.zeros(self.dims)
        self.ub      =  9 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('maze-random-10x10-plus-v0', enable_render=False)
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (4, 2)
        self.action_dict = ['N', 'E', 'S', 'W']
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 1
        self.leaf_size    = 20
        self.kernel_type  = "rbf"
        self.gamma_type   = "scale"
        self.ninits       = 30
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        M = x.reshape(self.policy_shape)
        
        returns = []
        observations = []
        actions = []
        
        for i in range(self.num_rollouts):
            obs    = self.env.reset()
            done   = False
            totalr = 0.
            steps  = 0
            while not done:
                # M      = self.policy
                inputs = (obs - self.mean)/self.std
                
                action = self.action_dict[np.argmax(np.dot(M, inputs))]
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps  += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)
            
        return np.mean(returns)*-1