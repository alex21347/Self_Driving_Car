# misc functions
import os
import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import time
import numpy as np
#from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev
import pandas as pd
import sys
from cone import *

sys.path.append(os.path.abspath(os.path.join('..', 'Moving_Car')))
def load_existing_map(name):
    
    left_cones = []
    right_cones = []

    current_dir = os.getcwd()
    map_path = os.path.join(current_dir, f"levels generated/{name}.csv")
    map_file = pd.read_csv(map_path)
    
    for i in range(len(map_file.iloc[:,0])):
        if map_file['Cone_Type'].iloc[i] == 'LEFT':

            left_cone = Cone(map_file['Cone_X'].iloc[i],map_file['Cone_Y'].iloc[i], Side.LEFT)
            left_cones.append(left_cone)
            # mouse_pos_list.append((map_file['Cone_X'].iloc[i]*ppu,map_file['Cone_Y'].iloc[i]*ppu))             
        
        else:
            right_cone = Cone(map_file['Cone_X'].iloc[i],map_file['Cone_Y'].iloc[i], Side.RIGHT)
            right_cones.append(right_cone)
            # mouse_pos_list.append((map_file['Cone_X'].iloc[i]*ppu,map_file['Cone_Y'].iloc[i]*ppu))             
        
    
    return left_cones, right_cones


def save_map(left_cones, right_cones):
    cone_x = []
    cone_y = []
    cone_type = []
    print('SAVE MAP AS : ')
    name = input()

    for i in range(len(left_cones)):
        cone_x.append(left_cones[i].position.x)
        cone_y.append(left_cones[i].position.y)
        cone_type.append('LEFT')
        
    for i in range(len(right_cones)):
        cone_x.append(right_cones[i].position.x)
        cone_y.append(right_cones[i].position.y)
        cone_type.append('RIGHT')       
        

    map_file = pd.DataFrame({'Cone_Type' : cone_type,
                                 'Cone_X' : cone_x,
                                 'Cone_Y' : cone_y})

    map_file.to_csv(f'levels generated/{name}.csv')


def load_map(mouse_pos_list, ppu):
    
    left_cones = []
    right_cones = []
    print('LOAD MAP : ')
    name = input()
    
    current_dir = os.getcwd()
    map_path = os.path.join(current_dir, f"levels generated/{name}.csv")
    map_file = pd.read_csv(map_path)
    
    for i in range(len(map_file.iloc[:,0])):
        if map_file['Cone_Type'].iloc[i] == 'LEFT':

            left_cone = Cone(map_file['Cone_X'].iloc[i],map_file['Cone_Y'].iloc[i], 'left')
            left_cones.append(left_cone)
            mouse_pos_list.append((map_file['Cone_X'].iloc[i]*ppu,map_file['Cone_Y'].iloc[i]*ppu))             
        
        else:
            right_cone = Cone(map_file['Cone_X'].iloc[i],map_file['Cone_Y'].iloc[i], 'right')
            right_cones.append(right_cone)
            mouse_pos_list.append((map_file['Cone_X'].iloc[i]*ppu,map_file['Cone_Y'].iloc[i]*ppu))             
        
    return left_cones, right_cones, mouse_pos_list



def bound_angle_180(angle):
    # car angle between (-180,180)
    temp_sign = np.mod(angle,360)
    if temp_sign > 180:
        angle_sign = -1
    else:
        angle_sign = 1
        
    angle = np.mod(angle,180)*angle_sign
    
    if angle < 0:
        angle = -180 - angle

    return angle


def exponential_weighted_average(array):
    len_array = len(array)
    coefs = np.ones(len_array)

    for i in range(len_array):
        coefs[len_array - i - 1] = np.exp(-0.1*(i))

    coefs = coefs/sum(coefs)
    val = sum(coefs * array)
    return val 

def cart_to_polar(array,car_angle):
    '''
    Assuming array is of relative-to-car x,y pairs
    '''

    polar_array = np.array((2,len(array)))
    for i in range(len(array)): 
        dist = np.linalg.norm(Vector2(array[0,i], array[1,i]))
        
        #calculating angle between car angle and sample point (alpha)
        a_b = Vector2(array[0,i], array[1,i])
        a_b = np.transpose(np.matrix([a_b.x,-1*a_b.y ]))
        rotate = np.matrix([[np.cos(-car_angle*np.pi/180),-1*np.sin(-car_angle*np.pi/180)],
                            [np.sin(-car_angle*np.pi/180),np.cos(-car_angle*np.pi/180)]])
        a_b = rotate*a_b
        a = a_b[0]
        b = a_b[1]
        beta = np.arctan(b/a)*(180/np.pi)
        alpha = beta + 90*(b/np.abs(b))*np.abs((a/np.abs(a)) - 1)
        angle = alpha[0,0]

        polar_array[:,i] = np.array([dist, angle])


    return polar_array


def omega_calc(velocity, theta, car_length):
    if theta:
        print("Agent steering angle : ",theta[0])
        turning_radius = car_length / sin(radians(theta[0]))
        angular_velocity = velocity / turning_radius
    else:
        angular_velocity = 0

    return angular_velocity

    