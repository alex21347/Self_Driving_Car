import numpy as np
from pp_functions.utils import exponential_weighted_average

def calculate_reward(self, action):

    #initialise reward
    reward = 0

    #reward for surviving a timestep
    reward += 0.01

    #reward for finding a new cone
    for key in self.pp.cone.new_in_fov_cone_flag.keys():
        if self.pp.cone.new_in_fov_cone_flag[key]:
            reward += 25

    #reward for completing the track (and fast!)
    if self.pp.track_number_changed and self.pp.track_number > 0 and not self.lap_reward_reset:
        reward += 700 - 8 * self.lap_time_running
        print("Lap complete!         Lap reward : ", 700 - 8 * self.lap_time_running, "     Lap time : ", self.lap_time_running, "    Lap : ", self.pp.track_number, "         ",self.pp.LEVEL_ID , flush = True)
        self.lap_reward_reset = True

    #if self.pp.track_number == 2:
     #   reward += 1000 - 5 * self.episode_time_running

    #if np.linalg.norm((7 * self.pp.ppu, 10 * self.pp.ppu) - self.pp.car.position * self.pp.ppu) < 40 and int(self.episode_time_running) > 4:
     #   reward += 500 - 5 * self.episode_time_running

    #reward for smooth driving
    reward -= np.linalg.norm(exponential_weighted_average(self.steering_angle_moving_average) - action)

    #reward for crashing
    if self.pp.car.crashed:
        reward = -1000

    return reward/10