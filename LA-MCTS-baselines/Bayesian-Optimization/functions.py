# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import gym
import json
import os
import highway_env
from gym_minigrid.wrappers import *
import pybullet_envs



class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("inf")
        self.curt_best_x = None
        self.foldername = foldername
        try:
            os.mkdir(foldername)
        except OSError:
            print ("Creation of the directory %s failed" % foldername)
        else:
            print ("Successfully created the directory %s " % foldername)
        
    def dump_trace(self):
        trace_path = self.foldername + '/result' + str(len( self.results) )
        final_results_str = json.dumps(self.results)
        with open(trace_path, "a") as f:
            f.write(final_results_str[1:-1] + '\n')
            
    def track(self, result, x = None):
        if result < self.curt_best:
            self.curt_best = result
            self.curt_best_x = x
        print("")
        print("="*10)
        print("iteration:", self.counter, "total samples:", len(self.results) )
        print("="*10)
        print("current best f(x):", self.curt_best)
        print("current best x:", np.around(self.curt_best_x, decimals=1).tolist())
        self.results.append(self.curt_best)
        self.counter += 1
        if len(self.results) % 100 == 0:
            self.dump_trace()
        
class Levy:
    def __init__(self, dims=1):
        self.dims    = dims
        self.lb      = -10 * np.ones(dims)
        self.ub      =  10 * np.ones(dims)
        self.counter = 0
        print("####dims:", dims)
        self.tracker = tracker('Levy'+str(dims))

    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = []
        for idx in range(0, len(x)):
            w.append( 1 + (x[idx] - 1) / 4 )
        w = np.array(w)
        
        term1 = ( np.sin( np.pi*w[0] ) )**2;
        
        term3 = ( w[-1] - 1 )**2 * ( 1 + ( np.sin( 2 * np.pi * w[-1] ) )**2 );
        
        
        term2 = 0;
        for idx in range(1, len(w) ):
            wi  = w[idx]
            new = (wi-1)**2 * ( 1 + 10 * ( np.sin( np.pi* wi + 1 ) )**2)
            term2 = term2 + new
        
        result = term1 + term2 + term3

        self.tracker.track( result, x )

        return result
    
        
class Ackley:
    def __init__(self, dims=3):
        self.dims    = dims
        self.lb      = -5 * np.ones(dims)
        self.ub      = 10 * np.ones(dims)
        self.counter = 0
        self.tracker = tracker('Ackley'+str(dims))
        

    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        
        self.tracker.track( result, x )
        
        return result    
    
    
class Schwefel:
    def __init__(self, dims=10):
        self.dims      = dims
        self.lb        = -500 * np.ones(dims)
        self.ub        =  500 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker('Schwefel'+str(dims) )
        
        
    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        result = np.sum(-x * np.sin(np.sqrt(np.abs(x)))) + 418.9829*x.size
        
        self.tracker.track( result, x )
                        
        return result
    
class Easom:
    def __init__(self):
        dims = 2
        self.dims      = dims
        self.lb        = -100 * np.ones(dims)
        self.ub        =  100 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker('Easom'+str(dims) )
        
    def __call__(self, x):
        x = np.array(x)
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        
        result = -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)
        self.tracker.track( result, x )
                
        return result
    
    
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
        self.tracker   = tracker('Highway'+str(self.dims) )
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        x = np.array(x)
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
            
        result = np.mean(returns)*-1
        self.tracker.track( result, x )
        return result
    
class Swimmer:
    
    def __init__(self):
        self.policy_shape = (2, 8)
        self.mean         = 0
        self.std          = 1
        self.dims         = 16
        self.lb           = -1 * np.ones(self.dims)
        self.ub           =  1 * np.ones(self.dims)
        self.counter      = 0
        self.env          = gym.make('Swimmer-v2')
        self.num_rollouts = 3
        self.tracker   = tracker('Swimmer'+str(self.dims) )
        
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
        
        self.render = False
        
    
    def __call__(self, x):
        x = np.array(x)
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
                
                action = np.dot(M, (obs - self.mean)/self.std)
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1                
                if self.render:
                    self.env.render()            
            returns.append(totalr)
            
        result = np.mean(returns)*-1
        self.tracker.track( result, x )
        return result
    
    
class MiniGrid:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 1
                
        self.dims    = 21
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = ImgObsWrapper(StateBonus(gym.make('MiniGrid-Empty-5x5-v0')))
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (7, 3)
        self.tracker   = tracker('MiniGrid'+str(self.dims) )
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        x = np.array(x)
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
                inputs = inputs.transpose(2,0,1).reshape(3,-1)
                
                action = np.argmax(np.sum(np.dot(M, inputs), axis=1))
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps  += 1
                if self.render:
                    self.env.render()
            returns.append(totalr)
            
        result = np.mean(returns)*-1
        self.tracker.track( result, x )
        return result

class RaceCar:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 1
                
        self.dims    = 4
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('RacecarBulletEnv-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (2, 2)
        self.tracker   = tracker('RaceCar'+str(self.dims) )
        
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        x = np.array(x)
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
            
        result = np.mean(returns)*-1
        self.tracker.track( result, x )
        return result
    
class Pusher:
    
    def __init__(self):
        self.mean    = 0
        self.std     = 1
                
        self.dims    = 385
        self.lb      = -1 * np.ones(self.dims)
        self.ub      =  1 * np.ones(self.dims)
        self.counter = 0
        self.env     = gym.make('PusherBulletEnv-v0')
        self.num_rollouts = 3
        self.render  = False
        self.policy_shape = (7, 55)
        self.tracker   = tracker('Pusher'+str(self.dims) )
        
        print("===========initialization===========")
        print("mean:", self.mean)
        print("std:", self.std)
        print("dims:", self.dims)
        print("policy:", self.policy_shape )
            
    def __call__(self, x):
        x = np.array(x)
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
            
        result = np.mean(returns)*-1
        self.tracker.track( result, x )
        return result