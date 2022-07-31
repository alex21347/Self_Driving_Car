import numpy as np
import pygame
import pp_functions
from stable_baselines3 import PPO, A2C, DQN
from matplotlib import pyplot as plt
from enum import Enum
from CarEnv import CarEnv
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle5 as pickle
if __name__ == "__main__":

    model_name = "PPO-cont"
    ID = '1653154333-324943'
    time_steps = 1410000
    models_dir = f'models2/{model_name}-{ID}'

    env = CarEnv(mode = "cont", midpoint_obs = True, test = False, ROS = True)

    if model_name == 'PPO' or model_name == 'PPO-cont':
        model = PPO.load(f"{models_dir}/car_model_{time_steps}")
    elif model_name == 'A2C' or model_name == 'A2C-cont':
        model = A2C.load(f"{models_dir}/car_model_{time_steps}") 
    elif model_name == 'DQN' or model_name == 'DQN-cont':
        model = DQN.load(f"{models_dir}/car_model_{time_steps}")

    robot_speed = 1
    observation = env.reset()


    #range_to_check = np.arange(-31, 31, 1)
    #mean_actions = np.zeros((len(range_to_check), 3))
    #a = 0

    #for i in range_to_check:
        #actions = []
        #actions1 = []
        #time_start = time.time()

    #need code to isolate a single rotation of the lidar, to indicate 1 time step.


    #load cone_classifier (0 = small cone, 1 = big cone)
    filename = 'cone_classifier_rf.sav'
    cone_classifier = pickle.load(open(filename, 'rb'))

    done = False
    
    while not done:
        #time_running = time.time() - time_start

        #incoming ROS message

        cone_info = {'left' : [[0.7,5+15], [0.3,45]], 'right' : [[0.7,5-15], [0.3,-45]]}
        #cone_info = {'left' : [], 'right' : []}

        #retrieve information from simulation
        observation, spline_error, midpoint_steering_angle = env.pp.get_observation_ros(cone_info, env.num_obs, midpoint_obs = env.midpoint_obs)
        if not spline_error:
            action = model.predict(observation)
            action = env.pp.car.max_steering * action[0]
        else:
            action = 0

        #actions.append(action)
        #actions1.append(midpoint_steering_angle)

        #construct new message for robot
        omega = pp_functions.utils.omega_calc(1, action, 2)
        robot_action = [robot_speed,omega]
        new_message = robot_action 

        #render simulation
        #pp_functions.drawing.render(env.pp)

        #SEND new_message TO ROBOT
        
        if env.pp.exit:
            break



    pygame.quit()


#    print(mean_actions)
    #plt.figure(figsize = (8,8))

    #plt.plot(mean_actions[:, 0], mean_actions[:, 1], label = 'Agent')
    #plt.plot(mean_actions[:, 0], mean_actions[:, 2], label = 'Midpoint')
    #plt.xlabel("angle of track")
    #plt.ylabel("steering angle")
    #plt.legend()
    #plt.show()