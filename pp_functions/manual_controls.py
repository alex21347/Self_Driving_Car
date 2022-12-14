import pygame
from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import time
import numpy as np
#from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev

import os
import sys
sys.path.append(os.path.abspath(os.path.join('..', '')))
from cone import *

import pp_functions.utils

def enable_dragging_screen(pp, events):
  #dragging screen using left mouse butto
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 or pp.moving_view_offset == True:
            if pp.moving_view_offset == False:
                pp.moving_view_offset = True
                pp.view_offset_mouse_pos_start = pygame.mouse.get_pos()
            mouse_pos = pygame.mouse.get_pos()
            mouseDelta = [float(mouse_pos[0] - pp.view_offset_mouse_pos_start[0]), float(mouse_pos[1] - pp.view_offset_mouse_pos_start[1])]
            pp.view_offset[0] = pp.prev_view_offset[0] + mouseDelta[0]
            pp.view_offset[1] = pp.prev_view_offset[1] + mouseDelta[1]

    for event in events:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pp.prev_view_offset[0] = pp.view_offset[0]
            pp.prev_view_offset[1] = pp.view_offset[1]
            pp.moving_view_offset = False


 # User input
def user_input(pp, events, dt):
    
    pressed = pygame.key.get_pressed()
                   
    #manual steering  
    if pressed[pygame.K_RIGHT]:
        pp.car.steering_angle -= 50 * dt
    elif pressed[pygame.K_LEFT]:
       pp.car.steering_angle += 50 * dt
    else:
        if(pp.car.steering_angle > (50 * dt)):
            pp.car.steering_angle -= 120 * dt
        elif(pp.car.steering_angle < -(50 * dt)):
            pp.car.steering_angle += 120 * dt
        else:
            pp.car.steering_angle = 0
                
    # press l for left cone
    if pressed[pygame.K_l]:
        mouse_pos = (pygame.mouse.get_pos()[0] - pp.view_offset[0], pygame.mouse.get_pos()[1] - pp.view_offset[1])
        
        if mouse_pos in pp.mouse_pos_list:
            pass
        else:
            
            make_cone = True
            for i in range(len(pp.mouse_pos_list)):
                if np.linalg.norm(tuple(x-y for x,y in zip(pp.mouse_pos_list[i],mouse_pos))) < 50:
                    make_cone = False
                    break
            
            if make_cone == True:
                left_cone = Cone(mouse_pos[0]/pp.ppu, mouse_pos[1]/pp.ppu, 'left')
                pp.cone.cone_list[Side.LEFT].append(left_cone)
                pp.mouse_pos_list.append(mouse_pos)
                
                
            
    # press r for right cone
    if pressed[pygame.K_r]:
        mouse_pos = (pygame.mouse.get_pos()[0] - pp.view_offset[0], pygame.mouse.get_pos()[1] - pp.view_offset[1])

        if mouse_pos in pp.mouse_pos_list:
            pass
        else:
            
            make_cone = True
            for i in range(len(pp.mouse_pos_list)):
                if np.linalg.norm(tuple(x-y for x,y in zip(pp.mouse_pos_list[i],mouse_pos))) < 50:
                    make_cone = False
                    break
            
            if make_cone == True:
                right_cone = Cone(mouse_pos[0]/pp.ppu, mouse_pos[1]/pp.ppu, 'right')
                pp.cone.cone_list[Side.RIGHT].append(right_cone)
                pp.mouse_pos_list.append(mouse_pos)
    
    
    
    #if CTRL + c then clear screen
    if pressed[pygame.K_LCTRL] and pressed[pygame.K_c]:
        #resetting most vars
        pp.target.targets  = []
        pp.target.non_passed_targets = []
        pp.cone.cone_list[Side.LEFT] = []
        pp.cone.cone_list[Side.RIGHT] = []    
        pp.cone.visible_cone_list[Side.LEFT] = []
        pp.cone.visible_cone_list[Side.RIGHT] = []
        pp.cone.in_fov_cone_list[Side.LEFT] = []
        pp.cone.in_fov_cone_list[Side.RIGHT] = []
        pp.path.splines[Side.LEFT] = []
        pp.path.splines[Side.RIGHT] = []
        pp.path.spline_linked[Side.RIGHT] == False
        pp.path.spline_linked[Side.LEFT] == False
        pp.mouse_pos_list = []
        pp.path.splines[Side.LEFT] = 0
        pp.path.splines[Side.RIGHT] = 0
        pp.cone.first_visible_cone[Side.LEFT] = 0
        pp.cone.first_visible_cone[Side.RIGHT] = 0
        pp.cone.first_cone_found[Side.RIGHT] = False
        pp.cone.first_cone_found[Side.LEFT] = False
        pp.track_number_changed = False
        pp.car.crashed = False
        pp.total_reward = 0
        
        
    #if 2 is pressed, increasing cruising speed
    #if 1 is pressed, decrease cruising speed
    
    for event in events:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
            pp.cruising_speed -= 0.05
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_2:
            pp.cruising_speed += 0.05

    #if a pressed then toggle automatic driving
    for event in events:
        if event.type == pygame.KEYUP and event.key == pygame.K_a: 
            if pp.car.auto == False:
                pp.car.auto  = True
            else:
                pp.car.auto  = False
    
                
    #if f pressed then toggle pp.fullscreen
    for event in events:
        if event.type == pygame.KEYUP and event.key == pygame.K_f:
            if pp.fullscreen == False:
                pp.fullscreen  = True
            else:
                pp.fullscreen  = False
        
        
    #if t pressed then set to pp.track mode
    for event in events:
        if event.type == pygame.KEYUP and event.key == pygame.K_t: 
            if pp.track == False:
                pp.track = True
            else:
                pp.track = False

    #if D then load map
    if  pressed[pygame.K_d]:
        
        #resetting most vars before loading
        pp.target.targets  = []
        pp.target.non_passed_targets = []
        pp.cone.cone_list[Side.LEFT] = []
        pp.cone.cone_list[Side.RIGHT] = []    
        pp.cone.visible_cone_list[Side.LEFT] = []
        pp.cone.visible_cone_list[Side.RIGHT] = []
        pp.cone.in_fov_cone_list[Side.LEFT] = []
        pp.cone.in_fov_cone_list[Side.RIGHT] = []
        pp.path.splines[Side.LEFT] = []
        pp.path.splines[Side.RIGHT] = []
        pp.path.spline_linked[Side.RIGHT] == False
        pp.path.spline_linked[Side.LEFT] == False
        pp.mouse_pos_list = []
        pp.path.splines[Side.LEFT] = 0
        pp.path.splines[Side.RIGHT] = 0
        pp.cone.first_visible_cone[Side.LEFT] = 0
        pp.cone.first_visible_cone[Side.RIGHT] = 0
        pp.cone.first_cone_found[Side.RIGHT] = False
        pp.cone.first_cone_found[Side.LEFT] = False
        pp.track_number_changed = False
        pp.car.crashed = False
        pp.total_reward = 0
        
        pp.cone.cone_list[Side.LEFT], pp.cone.cone_list[Side.RIGHT], pp.mouse_pos_list = pp_functions.utils.load_map(pp.mouse_pos_list, pp.ppu)
                
                
    #if S then save map
    if  pressed[pygame.K_s]:
        pp_functions.utils.save_map(pp.cone.cone_list[Side.LEFT], pp.cone.cone_list[Side.RIGHT])
        
        
    #dragging screen using left mouse butto
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 or pp.moving_view_offset == True:
            if pp.moving_view_offset == False:
                pp.moving_view_offset = True
                pp.view_offset_mouse_pos_start = pygame.mouse.get_pos()
            mouse_pos = pygame.mouse.get_pos()
            mouseDelta = [float(mouse_pos[0] - pp.view_offset_mouse_pos_start[0]), float(mouse_pos[1] - pp.view_offset_mouse_pos_start[1])]
            pp.view_offset[0] = pp.prev_view_offset[0] + mouseDelta[0]
            pp.view_offset[1] = pp.prev_view_offset[1] + mouseDelta[1]

    for event in events:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            pp.prev_view_offset[0] = pp.view_offset[0]
            pp.prev_view_offset[1] = pp.view_offset[1]
            pp.moving_view_offset = False

    #if CTRL + Z pressed then undo last left and right cone
    if pp.undo_done == False and pressed[pygame.K_LCTRL] and pressed[pygame.K_z]:
        pp.undo_done = True
        for side in Side:
          if len(pp.cone.visible_cone_list[side]) > 0:
              if pp.cone.cone_list[side][-1] == pp.cone.visible_cone_list[side][-1]:
                  pp.mouse_pos_list.remove((pp.cone.cone_list[side][-1].position.x * pp.ppu, pp.cone.cone_list[side][-1].position.y * pp.ppu))
                  pp.cone.cone_list[side].pop(-1)
                  pp.cone.visible_cone_list[side].pop(-1)
              else:
                  pp.mouse_pos_list.remove((pp.cone.cone_list[side][-1].position.x * pp.ppu, pp.cone.cone_list[side][-1].position.y * pp.ppu))
                  pp.cone.cone_list[side].pop(-1)
          else:
              if len(pp.cone.cone_list[side]) > 0:
                  pp.mouse_pos_list.remove((pp.cone.cone_list[side][-1].position.x * pp.ppu, pp.cone.cone_list[side][-1].position.y * pp.ppu))
                  pp.cone.cone_list[side].pop(-1)


    for event in events:
        if event.type == pygame.KEYUP and event.key == pygame.K_z:
            pp.undo_done = False


    #manual acceleration
    if pressed[pygame.K_UP]:
        if pp.car.velocity.x < 0:
            pp.car.acceleration = pp.car.brake_deceleration
        else:
            pp.car.acceleration += 1 * dt
    elif pressed[pygame.K_DOWN] and pp.car.breaks == True:
        if pp.car.velocity.x > 0:
            pp.car.acceleration = -pp.car.brake_deceleration
        else:
            pp.car.acceleration -= 1 * dt
    elif pressed[pygame.K_SPACE]:
        if abs(pp.car.velocity.x) > dt * pp.car.brake_deceleration:
            pp.car.acceleration = -copysign(pp.car.brake_deceleration, pp.car.velocity.x)
        else:
            pp.car.acceleration = -pp.car.velocity.x / dt
    else:
        if abs(pp.car.velocity.x) > dt * pp.car.free_deceleration:
            pp.car.acceleration = -copysign(pp.car.free_deceleration, pp.car.velocity.x)
        else:
            if dt != 0:
                pp.car.acceleration = -pp.car.velocity.x / dt
                
    # =============================================================================
#     
#     #If t pressed then create target
#     if pressed[pygame.K_t]
#             mouse_pos = pygame.mouse.get_pos()
#             
#             if mouse_pos in pp.mouse_pos_list:
#                 pass
#             else:
#                 make_target = True
#                 for i in range(len(pp.mouse_pos_list)):
#                     if np.linalg.norm(tuple(x-y for x,y in zip(pp.mouse_pos_list[i],mouse_pos))) < 25:
#                         make_target = False
#                         break
#                 
#                 if make_target == True:
#                     
#                     target = Target(mouse_pos[0]/pp.ppu,mouse_pos[1]/pp.ppu)
#                     targets.append(target)
#                     non_passed_targets.append(target)
#                     
#                     pp.mouse_pos_list.append(mouse_pos)
# =============================================================================
                