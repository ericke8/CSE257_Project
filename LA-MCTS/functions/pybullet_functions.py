import numpy as np
import gym
import json
import os
import pybullet_envs


class Minitaur:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 1
                
        self.dims    = 224
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('MinitaurBulletEnv-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (8, 28)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 10
        self.leaf_size    = 100
        self.kernel_type  = "linear"
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
    
    
class Kuka:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 10
                
        self.dims    = 27
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('KukaBulletEnv-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (3, 9)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 10
        self.leaf_size    = 100
        self.kernel_type  = "linear"
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
    
class Thrower:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 10
                
        self.dims    = 336
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('ThrowerBulletEnv-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (7, 48)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 10
        self.leaf_size    = 100
        self.kernel_type  = "linear"
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
    
class Pusher:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 10
                
        self.dims    = 385
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('PusherBulletEnv-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (7, 55)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 10
        self.leaf_size    = 100
        self.kernel_type  = "linear"
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
    
class FlagRun:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 10
                
        self.dims    = 748
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('HumanoidFlagrunBulletEnv-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (17, 44)
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp           = 10
        self.leaf_size    = 100
        self.kernel_type  = "linear"
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