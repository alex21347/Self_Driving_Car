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
import rospy 
from cone_tracker.msg import Cones
from geometry_msgs.msg import Twist
from numpysocket import NumpySocket
import socket
localIP     = "192.168.1.4"
localPort   = 20001
bufferSize  = 512
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
serverAddressPort   = ("192.168.1.47", 20002)
#UDPClientSocket.connect(serverAddressPort)
#npSocket = NumpySocket()
#npSocket.startClient('192.168.1.47', 1026)#TODO
env = CarEnv(mode = "cont", midpoint_obs = True, test = False, ROS = True)
model_name = "PPO-cont"
Method_RL = True
OIL = False
most_recent_cone_msg = None
#if Method_RL:
ID = '0-1'
time_steps = 1010000
#else:
#    ID = '1653154333-324943' #'0-1'
#    time_steps = 1040000  #1010000

models_dir = f'models2/{model_name}-{ID}'


#if model_name == 'PPO' or model_name == 'PPO-cont':
#    model = PPO.load(f"{models_dir}/car_model_{time_steps}")
#elif model_name == 'A2C' or model_name == 'A2C-cont':
#    model = A2C.load(f"{models_dir}/car_model_{time_steps}") 
#elif model_name == 'DQN' or model_name == 'DQN-cont':
#    model = DQN.load(f"{models_dir}/car_model_{time_steps}")

robot_speed = 1
observation = env.reset()


#load cone_classifier (0 = small cone, 1 = big cone)
filename1 = 'cone_classifier_rf.sav'
cone_classifier = pickle.load(open(filename1, 'rb'))

filename2 = 'OIL_model_rf.sav'
OIL_action_model = pickle.load(open(filename2, 'rb'))

publisher = rospy.Publisher("/robot_1/cmd_vel", Twist, queue_size=1)

def velocity_msg(omega):
    msg = Twist()
    msg.linear.x = 0.15
    msg.angular.z = omega
    return msg

last_msg = velocity_msg(0)

def cone_callback(message):
    global most_recent_cone_msg
    most_recent_cone_msg = message
    #timer_callback(0)

def timer_callback(asd):
    global most_recent_cone_msg
    if most_recent_cone_msg == None:
        return
    #cone_info = {'left' : [[0.7,5+15], [0.3,45]], 'right' : [[0.7,5-15], [0.3,-45]]}
    global last_msg
    left = []
    right = []
    for cone in most_recent_cone_msg.left_cones:
        left.append([cone.r, cone.theta * 360 / (2 * np.pi)])
    for cone in most_recent_cone_msg.right_cones:
        right.append([cone.r, cone.theta * 360 / (2 * np.pi)])
    cone_info = {'left' : left, 'right' : right}
    print(cone_info)

    #retrieve information from simulation
    observation, spline_error, midpoint_steering_angle = env.pp.get_observation_ros(cone_info, env.num_obs, midpoint_obs = env.midpoint_obs)
    print(observation.dtype)
    print(observation)
    observationMessage = observation.tobytes()
    UDPClientSocket.sendto(observationMessage, serverAddressPort)
    return
    #print(observation)
    #npSocket.send(observation)



    
    #if not spline_error:



        #action = model.predict(observation)
        #action = env.pp.car.max_steering * action[0]
    #else:
        #action = 0

    if OIL:
      action = OIL_action_model.predict(np.array([observation]))
      action = 1 * np.interp(action, [-1, 1], [-120, 120])

      if spline_error:
        action = 0.0


    action = midpoint_steering_angle
    #construct new message for robot
    omega = pp_functions.utils.omega_calc(1, [action], 2)
    msg = velocity_msg(omega)
    #robot_action = [robot_speed,omega]
    #new_message = robot_action 

    #SEND new_message TO ROBOT
    #if msg.angular.z != 0.0:
    #    publisher.publish(msg)
    #    last_msg = msg
    #else:
    #    publisher.publish(last_msg)
    #print(message)

    publisher.publish(msg)

rospy.init_node("controller")

timer = rospy.Timer(rospy.Duration(1.0 / 7.0), timer_callback)

if __name__ == "__main__":


    time.sleep(0.2)
    subscriber = rospy.Subscriber("/cones", Cones, cone_callback, queue_size=1)



        #incoming ROS message
        


    rospy.spin()
    #pygame.quit()
