import numpy as np
import gym
from gym.wrappers import FilterObservation, FlattenObservation
import pybullet_envs

env = gym.make('HumanoidFlagrunBulletEnv-v0')
# env = gym.make('Hopper-v2')


env.render()

# env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
obs = env.reset()

print('action space')
print(env.action_space)

print('obs space')
print(env.observation_space)

print('obs high, low')
print(env.observation_space.high)
print(env.observation_space.low)

print('obs shape')
print(obs.shape)

done = False

def policy():
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()

    
for i in range(2000):
    env.render()
    action = policy()
    obs, reward, done, info = env.step(action)

    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    #substitute_goal = obs['achieved_goal'].copy()
    #substitute_reward = env.compute_reward(
        #obs['achieved_goal'], substitute_goal, info)
    #print('reward is {}, substitute_reward is {}'.format(
        #reward, substitute_reward))
