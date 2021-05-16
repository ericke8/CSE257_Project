# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import numpy as np
import gym
import json
import os

import imageio


class tracker:
    def __init__(self, foldername):
        self.counter   = 0
        self.results   = []
        self.curt_best = float("inf")
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
            f.write(final_results_str + '\n')
            
    def track(self, result):
        if result < self.curt_best:
            self.curt_best = result
        self.results.append(self.curt_best)
        if len(self.results) % 100 == 0:
            self.dump_trace()

class Levy:
    def __init__(self, dims=10):
        self.dims        = dims
        self.lb          = -10 * np.ones(dims)
        self.ub          =  10 * np.ones(dims)
        self.tracker     = tracker('Levy'+str(dims))
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 10
        self.leaf_size   = 8
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type   = "auto"
        print("initialize levy at dims:", self.dims)
        
    def __call__(self, x):
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
        self.tracker.track( result )

        return result

class Ackley:
    def __init__(self, dims=10):
        self.dims      = dims
        self.lb        = -5 * np.ones(dims)
        self.ub        =  10 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker('Ackley'+str(dims) )
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 1
        self.leaf_size = 10
        self.ninits    = 40
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e )
        self.tracker.track( result )
                
        return result

class Lunarlanding:
    def __init__(self):
        self.dims = 12
        self.lb   = np.zeros(12)
        self.ub   = 2 * np.ones(12)
        self.counter = 0
        self.env = gym.make('LunarLander-v2')
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 50
        self.leaf_size   = 10
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type  = "scale"
        
        self.render      = False
        
        
    def heuristic_Controller(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
    
        total_rewards = []
        for i in range(0, 3): # controls the number of episode/plays per trial
            state = self.env.reset()
            rewards_for_episode = []
            num_steps = 2000
        
            for step in range(num_steps):
                if self.render:
                    self.env.render()
                received_action = self.heuristic_Controller(state, x)
                next_state, reward, done, info = self.env.step(received_action)
                rewards_for_episode.append( reward )
                state = next_state
                if done:
                     break
                        
            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append( np.sum(rewards_for_episode) )
        total_rewards = np.array(total_rewards)
        mean_rewards = np.mean( total_rewards )
        
        return mean_rewards*-1


#############################################################################
class Schwefel:
    def __init__(self, dims=10):
        self.dims      = dims
        self.lb        = -500 * np.ones(dims)
        self.ub        =  500 * np.ones(dims)
        self.counter   = 0
        self.tracker   = tracker('Schwefel'+str(dims) )
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp        = 1
        self.leaf_size = 10
        self.ninits    = 40
        self.kernel_type = "rbf"
        self.gamma_type  = "auto"
        
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        result = np.sum(-x * np.sin(np.sqrt(np.abs(x)))) + 418.9829*x.size
        self.tracker.track( result )
                
        return result


class LunarlandingCont:
    def __init__(self):
        self.dims = 12
        self.lb   = np.zeros(12)
        self.ub   = 2 * np.ones(12)
        self.counter = 0
        self.env = gym.make('LunarLanderContinuous-v2')
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 50
        self.leaf_size   = 10
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type  = "scale"
        
        self.render      = False
        
        
    def heuristic_Controller(self, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]
        

        a = np.array([hover_todo*w[10] - w[11], -angle_todo*w[10]])
        a = np.clip(a, -1, +1)
        # a = 0
        # if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
        #     a = 2
        # elif angle_todo < -w[11]:
        #     a = 3
        # elif angle_todo > +w[11]:
        #     a = 1
        return a
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
    
        total_rewards = []
        for i in range(0, 3): # controls the number of episode/plays per trial
            state = self.env.reset()
            rewards_for_episode = []
            num_steps = 2000
        
            for step in range(num_steps):
                if self.render:
                    self.env.render()
                received_action = self.heuristic_Controller(state, x)
                next_state, reward, done, info = self.env.step(received_action)
                rewards_for_episode.append( reward )
                state = next_state
                if done:
                     break
                        
            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append( np.sum(rewards_for_episode) )
        total_rewards = np.array(total_rewards)
        mean_rewards = np.mean( total_rewards )
        
        return mean_rewards*-1


class Walker:
    def __init__(self):
        self.dims = 16
        self.lb   = -20 * np.ones(16)
        self.ub   = 20 * np.ones(16)
        self.counter = 0
        self.env = gym.make('BipedalWalker-v3')
        
        #tunable hyper-parameters in LA-MCTS
        self.Cp          = 50
        self.leaf_size   = 10
        self.kernel_type = "poly"
        self.ninits      = 40
        self.gamma_type  = "scale"
        
        self.render      = False

        self.SUPPORT_KNEE_ANGLE = +0.1
        
        
    def heuristic_Controller(self, s, w, walk_state_dict):
        SPEED = w[0]  # Will fall forward on higher speed
        STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3

        SUPPORT_KNEE_ANGLE = self.SUPPORT_KNEE_ANGLE

        moving_leg = walk_state_dict['moving_leg']
        supporting_leg = walk_state_dict['supporting_leg']
        walk_state = walk_state_dict['state']
        supporting_knee_angle = walk_state_dict['supporting_knee_angle']

        contact0 = s[8]
        contact1 = s[13]
        moving_s_base = 4 + 5*moving_leg
        supporting_s_base = 4 + 5*supporting_leg

        hip_targ  = [None,None]   # -0.8 .. +1.1
        knee_targ = [None,None]   # -0.6 .. +0.9
        hip_todo  = [0.0, 0.0]
        knee_todo = [0.0, 0.0]

        if walk_state==STAY_ON_ONE_LEG:
            hip_targ[moving_leg]  = w[1]
            knee_targ[moving_leg] = w[2]
            supporting_knee_angle += w[3]
            if s[2] > SPEED: supporting_knee_angle += w[4]
            supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[supporting_s_base+0] < w[5]: # supporting leg is behind
                walk_state_dict['state'] = PUT_OTHER_DOWN
        if walk_state==PUT_OTHER_DOWN:
            hip_targ[moving_leg]  = w[6]
            knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
            knee_targ[supporting_leg] = supporting_knee_angle
            if s[moving_s_base+4]:
                walk_state_dict['state'] = PUSH_OFF
                supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
        if walk_state==PUSH_OFF:
            knee_targ[moving_leg] = supporting_knee_angle
            knee_targ[supporting_leg] = w[7]
            if s[supporting_s_base+2] > w[8] or s[2] > 1.2*SPEED:
                walk_state_dict['state'] = STAY_ON_ONE_LEG
                walk_state_dict['moving_leg'] = 1 - moving_leg
                walk_state_dict['supporting_leg'] = 1 - moving_leg

        if hip_targ[0]: hip_todo[0] = w[9]*(hip_targ[0] - s[4]) - w[10]*s[5]
        if hip_targ[1]: hip_todo[1] = w[9]*(hip_targ[1] - s[9]) - w[10]*s[10]
        if knee_targ[0]: knee_todo[0] = w[11]*(knee_targ[0] - s[6])  - w[12]*s[7]
        if knee_targ[1]: knee_todo[1] = w[11]*(knee_targ[1] - s[11]) - w[12]*s[12]

        hip_todo[0] -= w[13]*(0-s[0]) - w[14]*s[1] # PID to keep head strait
        hip_todo[1] -= w[13]*(0-s[0]) - w[14]*s[1]
        knee_todo[0] -= w[15]*s[3]  # vertical speed, to damp oscillations
        knee_todo[1] -= w[15]*s[3]

        a = np.array([0.0, 0.0, 0.0, 0.0])
        a[0] = hip_todo[0]
        a[1] = knee_todo[0]
        a[2] = hip_todo[1]
        a[3] = knee_todo[1]
        a = np.clip(0.5*a, -1.0, 1.0)
        return a, walk_state_dict
        
    def __call__(self, x):
        self.counter += 1
        assert len(x) == self.dims
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
    
        total_rewards = []
        for i in range(0, 3): # controls the number of episode/plays per trial
            state = self.env.reset()
            walk_state_dict = {
                'state': 1,
                'moving_leg': 0,
                'supporting_leg': 1,
                'supporting_knee_angle': self.SUPPORT_KNEE_ANGLE
            }
            rewards_for_episode = []
            num_steps = 2000
            done = False
            while not done:
                if self.render:
                    self.env.render()
                received_action, walk_state_dict = self.heuristic_Controller(state, x, walk_state_dict)
                next_state, reward, done, info = self.env.step(received_action)
                rewards_for_episode.append( reward )
                state = next_state
                if done:
                     break
                        
            rewards_for_episode = np.array(rewards_for_episode)
            total_rewards.append( np.sum(rewards_for_episode) )
        total_rewards = np.array(total_rewards)
        mean_rewards = np.mean( total_rewards )
        
        return mean_rewards*-1
