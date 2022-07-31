import gym
import numpy as np
import time
import pygame
import pp_functions
from pp_functions.reward_function import calculate_reward
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib import pyplot as plt

from CarEnv import CarEnv

if __name__ == "__main__":

    episodes = 7

    actions = []
    speeds = []
    rewards = []
    observations = []

    for i in range(episodes):
        env = CarEnv(mode = "cont", midpoint_obs = False, test = False, LEVEL_ID = f'MAP_{(i % 7) + 1}')
        done = False
        observation = env.reset()
        while not done:
            action = env.pp.midpoint_steering_angle()
            action = [np.interp(action, [-120, 120], [-1, 1])]
            actions.append(action[0])
            observation, reward, done, info = env.step(action)
            observations.append(observation)

            env.render()

            if done:
                speeds.append(env.lap_time_running)
                rewards.append(env.total_reward)
                print('Guide Policy total reward : ', env.total_reward)

            if env.pp.exit:
                break

    pygame.quit()

print(len(observations))
print(len(actions))
print(observations[10])
print(actions[10])

np.savetxt("observations_data.csv", observations, delimiter=",")
np.savetxt("actions_data.csv", actions, delimiter=",")