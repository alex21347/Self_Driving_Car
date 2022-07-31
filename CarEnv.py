import random

import gym
import numpy as np
import time
from path_planning import PathPlanning
import pygame
import pp_functions
from pp_functions.reward_function import calculate_reward
from stable_baselines3.common.env_checker import check_env

class CarEnv(gym.Env):
    def __init__(self,
                 mode = 'cont',
                 midpoint_obs = False,
                 test = False,
                 ROS = False,
                 LEVEL_ID = 'None',
                 random_guide = False,
                 robustness_test = False,
                 robustness_scale = 0.0):

        super(CarEnv, self).__init__()

        self.LEVEL_ID = LEVEL_ID
        self.ROS = ROS
        self.mode = mode
        self.num_envs = 1
        self.test = test
        self.random_guide = random_guide
        self.midpoint_obs = midpoint_obs
        self.robustness_test = robustness_test
        self.robustness_scale = robustness_scale

        self.episode_num = 0

        self.clock = pygame.time.Clock()

        self.num_steps = 0

        if self.LEVEL_ID == 'None':
            self.no_specific_map = True
        else:
            self.no_specific_map = False

        self.pp = PathPlanning(test = self.test, ROS = self.ROS, LEVEL_ID = self.LEVEL_ID)

        self.LEVEL_ID = self.pp.LEVEL_ID

        self.episode_time_start = time.time()
        self.lap_time_start = time.time()
        self.episode_time_running = 0
        self.lap_time_running = 0
        self.steering_angle_moving_average = np.zeros(20)
        self.total_reward = 0
        self.lap_reward_reset = False
        self.lap_reset = False

        self.data_logger = {'episode_end' : [], 'action_state_pair' : []}

        if self.mode == "cont":
            self.action_space = gym.spaces.Box(low = -1, high = 1, shape=(1,), dtype=np.float32)
        elif self.mode == "discrete":
            self.action_space = gym.spaces.Discrete(5)

        self.num_obs = 0 + int(midpoint_obs) + 5 * 2 * 2 # 0 car obs + guide policy action + 5 spline points * 2 (angle + dist) * 2 (left/right sides) (22)

        self.obs_log = []

        low = -1 * np.ones(self.num_obs)
        high = 1 * np.ones(self.num_obs)

        #low[:2] = -1
        #high[:2] = 1

        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(self.num_obs,), dtype=np.float32)

    def generate_random_action(self):
        car_steering_angle = random.uniform(-self.pp.car.max_steering, self.pp.car.max_steering)
        car_curr_velocity = self.pp.cruising_speed
        return [car_steering_angle, car_curr_velocity]

    def update_moving_average(self, new_element):
        len_array = len(self.steering_angle_moving_average)
        new_array = np.zeros(len_array)
        new_array[:len_array-1] = self.steering_angle_moving_average[1:]
        new_array[-1] = new_element
        self.steering_angle_moving_average = new_array

    def render(self, mode=None):
        pp_functions.drawing.render(self.pp)
        
    def step(self, action):

        self.num_steps += 1

        #this fixes a bug where the the cars first steering angle set the car angle, instead of an incremental change
        if self.mode == "cont": 
            if self.num_steps > 2:
                self.pp.car.steering_angle = self.pp.car.max_steering * action[0]
            else:
                self.pp.car.steering_angle = 0

        elif self.mode == "discrete":
            if action == 0:
                self.pp.car.steering_angle = 0.5 * self.pp.car.max_steering
            elif action == 1:
                self.pp.car.steering_angle = 0.5 * -1 * self.pp.car.max_steering
            elif action == 2:
                self.pp.car.steering_angle = 1  * self.pp.car.max_steering
            elif action == 3:
                self.pp.car.steering_angle = 1 * -1 * self.pp.car.max_steering
            elif action == 4:
                self.pp.car.steering_angle = 0 * self.pp.car.max_steering
            
        self.pp.car.velocity.x = 1
        
        dt = self.clock.get_time() / 500 

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.pp.exit = True
        pp_functions.manual_controls.enable_dragging_screen(self.pp, events)

        self.episode_time_running = time.time() - self.episode_time_start

        if not self.pp.track_number_changed:
            self.lap_reset = False
            self.lap_reward_reset = False

        if self.pp.track_number_changed and not self.lap_reset:
            self.lap_time_start = time.time()
            self.lap_reset = True

        self.lap_time_running = time.time() - self.lap_time_start

        self.pp.car.config_angle()

        # update target list
        self.pp.target.update_target_lists()
       
        # update cone list
        self.pp.cone.update_cone_list(self.pp)
        
        # calculate closest target
        self.pp.target.update_closest_target()

        # reset targets for new lap
        self.pp.reset_new_lap()

        # implement track logic
        self.pp.track_logic()

        #calculate midpoints for guide policy
        self.pp.path.generate_midpoint_path(self.pp)

        # Logic
        self.pp.implement_main_logic(dt)

        # Retrieve observation
        observation = self.pp.get_observation(self.num_obs, self.midpoint_obs, self.steering_angle_moving_average, self.random_guide, self.robustness_test, self.robustness_scale)

        done, episode_end = self.pp.set_done(self.episode_time_running, self.episode_num, self.num_steps)
        if episode_end:
            self.data_logger['episode_end'].append(episode_end)

        reward = calculate_reward(self, action[0])
        self.pp.reward = reward
        self.total_reward += reward

        #updating steering_angle_moving_average
        self.update_moving_average(action[0])

        info = {}

        self.clock.tick(self.pp.ticks)

        self.obs_log.append(observation)

        return observation, reward, done, info

    def reset(self):
        if self.no_specific_map:
            self.pp = PathPlanning(test = self.test, ROS = self.ROS, LEVEL_ID = 'None')
        else:
            self.pp = PathPlanning(test = self.test, ROS = self.ROS, LEVEL_ID = self.pp.LEVEL_ID)

        self.total_reward = 0
        self.num_steps = 0

        self.episode_num += 1
        
        self.episode_time_start = time.time()
        self.lap_time_start = time.time()

        observation = np.zeros(self.num_obs)
        return observation

if __name__ == "__main__":
    env = CarEnv()
    observation = env.reset()
    check_env(env)
    done = False

    while not done:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        env.render()

        if done:
            print(env.total_reward)

        if env.pp.exit:
            break

    pygame.quit()