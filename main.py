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
import pickle5 as pickle
from CarEnv import CarEnv



def load_model(models_dir, time_steps):
    if model_name == 'PPO' or model_name == 'PPO-cont':
        model = PPO.load(f"{models_dir}/car_model_{time_steps}")
    elif model_name == 'A2C' or model_name == 'A2C-cont':
        model = A2C.load(f"{models_dir}/car_model_{time_steps}") 
    elif model_name == 'DQN' or model_name == 'DQN-cont':
        model = DQN.load(f"{models_dir}/car_model_{time_steps}")
    return model

if __name__ == "__main__":

    episodes = 1

    # ----- RL + GUIDE POLICY (COIL) -----
    model_name = "PPO-cont"
    ID = '1653154333-324943' #'1657126263-575786' 
    time_steps = 1410000 #750000
    models_dir = f'models2/{model_name}-{ID}'
    model = load_model(models_dir, time_steps)
    
    actions1 = []
    speeds1 = []
    rewards1 = []
    survival1 = []

    for i in range(episodes):
        env = CarEnv(mode = "cont", midpoint_obs = True, test = True, LEVEL_ID = f'MAP_test{(i % 3) + 1}')
        done = False
        observation = env.reset()
        while not done:
            action, _states = model.predict(observation)
            actions1.append(action)
            observation, reward, done, info = env.step(action)
            env.render()

            if done:
                if env.data_logger['episode_end'][-1][0] != 'crash':
                    speeds1.append(env.lap_time_running)
                    survival1.append(1)
                else:
                    print('car crashed so not appending speed list')
                    survival1.append(0)

                rewards1.append(env.total_reward)
                print('COIL total reward : ', env.total_reward)

            if env.pp.exit:
                break

    stds1 = []
    for i in range(len(actions1) - 10):
        stds1.append(np.std(actions1[i:i+10]))


    # ----- RL ONLY -----
    model_name = "PPO-cont"
    ID = '1657126289-507969' #'1654378987-726874' 
    time_steps = 710000 #1010000
    models_dir = f'models3/{model_name}-{ID}'
    model = load_model(models_dir, time_steps)

    actions2 = []
    speeds2 = []
    rewards2 = []
    survival2 = []

    for i in range(episodes):
        env = CarEnv(mode = "cont", midpoint_obs = False, test = True, LEVEL_ID = f'MAP_test{(i % 3) + 1}')
        done = False
        observation = env.reset()
        while not done:
            action, _states = model.predict(observation)
            actions2.append(action)
            observation, reward, done, info = env.step(action)
            env.render()

            if done:
                if env.data_logger['episode_end'][-1][0] != 'crash':
                    speeds2.append(env.lap_time_running)
                    survival2.append(1)
                else:
                    print('car crashed so not appending speed list')
                    survival2.append(0)
                rewards2.append(env.total_reward)
                print('RL total reward : ', env.total_reward)

            if env.pp.exit:
                break

    stds2 = []
    for i in range(len(actions2) - 10):
        stds2.append(np.std(actions2[i:i+10]))


    # ----- GUIDE POLICY -----
    #actions3 = []
    #speeds3 = []
    #rewards3 = []
    #survival3 = []

    #for i in range(episodes):
        #env = CarEnv(mode = "cont", midpoint_obs = True, test = True, LEVEL_ID = f'MAP_test{(i % 3) + 1}')
        #done = False
        #observation = env.reset()
        #while not done:
            #action = env.pp.midpoint_steering_angle()
            #action = [np.interp(action, [-120, 120], [-1, 1])]
            ##action, _states = model.predict(observation)
            #actions3.append(action[0])
            #observation, reward, done, info = env.step(action)
            #env.render()

            #if done:
                #if env.data_logger['episode_end'][-1][0] != 'crash':
                    #speeds3.append(env.lap_time_running)
                    #survival3.append(1)
                #else:
                    #print('car crashed so not appending speed list')
                    #survival3.append(0)
                #rewards3.append(env.total_reward)
                #print('Guide Policy total reward : ', env.total_reward)

            #if env.pp.exit:
                #break

    #stds3 = []
    #for i in range(len(actions3) - 10):
        #stds3.append(np.std(actions3[i:i+10]))



    ##OIL

    #actions4 = []
    #speeds4 = []
    #rewards4 = []
    #survival4 = []

    #filename2 = 'OIL_model_rf.sav'
    #OIL_action_model = pickle.load(open(filename2, 'rb'))

    #for i in range(episodes):
        #env = CarEnv(mode = "cont", midpoint_obs = False, test = True, LEVEL_ID = f'MAP_test{(i % 3) + 1}')
        #done = False
        #observation = env.reset()
        #while not done:
            #action = OIL_action_model.predict(np.array([observation]))
            ##action = np.interp(action, [-1, 1], [-120, 120])
            #observation, reward, done, info = env.step(action)
            #env.render()
            #actions4.append(action[0])

            #if done:
                #if env.data_logger['episode_end'][-1][0] != 'crash':
                    #speeds4.append(env.lap_time_running)
                    #survival4.append(1)
                #else:
                    #print('car crashed so not appending speed list')
                    #survival4.append(0)

                #rewards4.append(env.total_reward)
                #print('OIL total reward : ', env.total_reward)

            #if env.pp.exit:
                #break

    #stds4 = []
    #for i in range(len(actions4) - 10):
            #stds4.append(np.std(actions4[i:i+10]))
        
    pygame.quit()


    #plt.figure(figsize = (10,8))
    #plt.plot(np.array(range(len(actions1))), np.array(actions1))
    #plt.show()

    #plt.figure(figsize = (10,8))
    #plt.plot( np.array(range(len(actions2))), np.array(actions2))
    #plt.show()

    #plt.figure(figsize = (10,8))
    #plt.plot( np.array(range(len(actions3))), np.array(actions3))
    #plt.show()

    #plt.figure(figsize = (10,8))
    #plt.hist(stds1, alpha = 0.5, label = "COIL", density = True)
    #plt.hist(stds2, alpha = 0.5, label = "RL", density = True)
    #plt.hist(stds3, alpha = 0.5, label = "Guide Policy", density = True)
    #plt.legend()
    #plt.show()

    def exponential_weighted_average(array):
        len_array = len(array)
        coefs = np.ones(len_array)

        for i in range(len_array):
            coefs[len_array - i - 1] = np.exp(-0.1*(i))

        coefs = coefs/sum(coefs)
        val = sum(coefs * array)
        return val 

    window_size = 20


    #OIL_vals = np.zeros(len(actions4) - window_size)
    #for i in range(window_size, len(actions4)):
        #OIL_vals[i - window_size] = np.linalg.norm(exponential_weighted_average(actions4[i-window_size:i]) -actions4[i])

    #guide_vals = np.zeros(len(actions3) - window_size)
    #for i in range(window_size, len(actions3)):
        #guide_vals[i - window_size] = np.linalg.norm(exponential_weighted_average(actions3[i-window_size:i]) - actions3[i])

    rl_vals = np.zeros(len(actions2) - window_size)
    for i in range(window_size, len(actions2)):
        rl_vals[i - window_size] = np.linalg.norm(exponential_weighted_average(actions2[i-window_size:i]) - actions2[i])

    COIL_vals = np.zeros(len(actions1) - window_size)
    for i in range(window_size, len(actions1)):
        COIL_vals[i - window_size] = np.linalg.norm(exponential_weighted_average(actions1[i-window_size:i]) - actions1[i])


    #print()
    print('----- Smoothness - std -----')
    print("COIL: ", np.mean(stds1))
    print("RL : ", np.mean(stds2))
   # print("Guide Policy : ", np.mean(stds3))
   # print("OIL : ", np.mean(np.array(stds4)))
   # print()
    print('----- Smoothness - expo weighted -----')
    print("COIL: ", np.sum(COIL_vals))
    print("RL : ", np.sum(rl_vals))
    print("COIL len: ", len(COIL_vals))
    print("RL len : ", len(rl_vals))
    #print("Guide Policy : ", np.mean(guide_vals))
    #print("OIL : ", np.mean(np.array(OIL_vals)))
    #print()
    print('----- Reward -----')
    print('COIL : ', np.mean(np.array(rewards1)))
    print('RL : ', np.mean(np.array(rewards2)))
    #print('Guide Policy', np.mean(np.array(rewards3)))
    #print('OIL : ', np.mean(np.array(rewards4)))
    #print()
    print('----- Lap time -----')
    print('COIL : ', np.sum(np.array(speeds1)))
    print('RL : ', np.sum(np.array(speeds2)))
   # print('Guide Policy', np.mean(np.array(speeds3)))
    #print('OIL : ', np.mean(np.array(speeds4)))
    #print()
    print('----- Survival Rate -----')
    print('COIL : ', np.mean(np.array(survival1)))
    print('RL : ', np.mean(np.array(survival2)))
    #print('Guide Policy', np.mean(np.array(survival3)))
    #print('OIL', np.mean(np.array(survival4)))



    #obs_log = env.obs_log

    #num_obs = env.num_obs

    #obs_log_1 = np.zeros((len(obs_log), num_obs))

    #for i in range(len(obs_log)):
        #for j in range(num_obs):
            #obs_log_1[i,j] = obs_log[i][j]

    #plt.figure(figsize=(10,8))
    #plt.boxplot(obs_log_1[:,:2])
    #plt.title('car stats')
    #plt.show()

    #plt.figure(figsize=(10,8))
    #plt.boxplot(obs_log_1[:,2::2])
    #plt.title('boundary sample dists')
    #plt.show()

    #plt.figure(figsize=(10,8))
    #plt.boxplot(obs_log_1[:,3::2])
    #plt.title('boundary sample angles')
    #plt.show()

    #summary_stats = np.zeros((num_obs, 2))

    #for i in range(num_obs):
        #summary_stats[i, 0] = np.mean(obs_log_1[:,i])
        #summary_stats[i, 1] = np.std(obs_log_1[:,i])

    #np.savetxt("observation_normalisation_statistics.csv", summary_stats, delimiter=",")