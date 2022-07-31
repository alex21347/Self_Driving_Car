import os
import pygame
import time
import random

from sklearn.preprocessing import robust_scale

from car import Car
from cone import *
from path import *

import pp_functions.manual_controls
import pp_functions.drawing
from pp_functions.reward_function import calculate_reward
from pp_functions.utils import exponential_weighted_average, bound_angle_180

class PathPlanning:
    def __init__(self, test = False, ROS = False, LEVEL_ID = 'None'):
        self.ROS = ROS
        self.target = Target(-1000,-1000) #could this be a an empty list instead?
        self.car = Car(7,10)
        self.cone = Cone(-1000,-1000,Side.LEFT) #could this be a an empty list instead?
        self.path = Path()
        self.LEVEL_ID = LEVEL_ID
        self.initialize_images()
        if not self.ROS:
            self.initialize_map(test) #comment this if you want to start with blank sheet map
        self.observation_statistics = None
        self.load_observation_statistics('observation_normalisation_statistics.csv')

        pygame.init()
        pygame.display.set_caption("Car")

        self.width = 1280
        self.height = 720
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.fullscreen = False
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.mouse_pos_list = []
        self.total_reward = 0
        self.cruising_speed = 1
        self.ppu = 32

        self.view_offset = [0, 0]
        self.car_centre = False #THIS DOESNT WORK YET
        self.prev_view_offset = [0, 0]
        self.moving_view_offset = False
        self.view_offset_mouse_pos_start = [0, 0]
        self.midpoint_created = False
        self.undo_done = False

        self.track = True
        self.track_number = -1
        self.track_number_changed = False
        self.time_start_sim = None
        
        self.current_midpoint_steering_angle = None
        
        self.time_running = 0
        self.reward = 0
        self.done = False

        self.episode_num = None
        self.num_steps = 0
        self.ROS = ROS

    def initialize_images(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "images/car_r_30.png")
        self.car.car_image = pygame.image.load(image_path)

        # image_path7 = os.path.join(current_dir, "explosion_image.png")
        # explosion_image = pygame.image.load(image_path7)
        
        image_path1 = os.path.join(current_dir, "images/target_r_t.png")
        self.target.image = pygame.image.load(image_path1)
        
        image_path3 = os.path.join(current_dir, "images/left_cone_s.png")
        self.cone.image[Side.LEFT] = pygame.image.load(image_path3)
        
        image_path4 = os.path.join(current_dir, "images/right_cone_s.png")
        self.cone.image[Side.RIGHT] = pygame.image.load(image_path4)

        image_path5 = os.path.join(current_dir, "images/left_spline_s.png")
        self.path.spline_image[Side.LEFT] = pygame.image.load(image_path5)
        
        image_path6 = os.path.join(current_dir, "images/right_spline_s.png")
        self.path.spline_image[Side.RIGHT] = pygame.image.load(image_path6)

    def initialize_map(self, test):

        if self.LEVEL_ID == 'None':
            if not test:
                random_number = random.randint(1, 7) #7 total
                self.LEVEL_ID = f"MAP_{random_number}"
            else:
                random_number = random.randint(1, 3) #3 total
                self.LEVEL_ID = f"MAP_test{random_number}"

        left_cones, right_cones = pp_functions.utils.load_existing_map(self.LEVEL_ID)
        self.cone.cone_list[Side.LEFT] = left_cones
        self.cone.cone_list[Side.RIGHT] = right_cones

    def reset_new_lap(self):
        # reset targets for new lap
        if self.track_number_changed:
        #if (len(self.target.targets) > 0
        #and len(self.target.non_passed_targets) == 0 
        #and (self.path.spline_linked[Side.LEFT] == True or self.path.spline_linked[Side.RIGHT] == True)
        #and self.track):
            self.target.reset_targets()

    def track_logic(self):
        if self.cone.first_visible_cone[Side.LEFT] != 0 and self.cone.first_visible_cone[Side.RIGHT] != 0: 
            
            #Setting the finishing line/point
            if not self.midpoint_created and self.track:
                self.path.start_midpoint_x = np.mean([self.cone.first_visible_cone[Side.LEFT].position.x, self.cone.first_visible_cone[Side.RIGHT].position.x])
                self.path.start_midpoint_y = np.mean([self.cone.first_visible_cone[Side.LEFT].position.y, self.cone.first_visible_cone[Side.RIGHT].position.y])     
                self.midpoint_created = True
                
            #Incrementing lap number by 1
            elif (np.linalg.norm((self.path.start_midpoint_x, self.path.start_midpoint_y)-self.car.position) < 20/self.ppu 
            and not self.track_number_changed and self.track):
                self.track_number += 1
                lap_reward = True
                self.track_number_changed = True
                
            #Setting track_number_changed to false when not on finishing line
            elif (np.linalg.norm((self.path.start_midpoint_x, self.path.start_midpoint_y)-self.car.position) > 20/self.ppu
            and self.track):
                self.track_number_changed = False  

    def steering(self):
        if (len(self.target.visible_targets) > 0 
        and np.linalg.norm(self.target.closest_target.position-self.car.position) < self.car.fov/self.ppu
        and np.linalg.norm(self.target.closest_target.position-self.car.position) > 20/self.ppu
        and self.car.auto == True 
        and self.target.closest_target.passed == False):
            
            dist = self.target.closest_target.dist_car
            alpha = self.target.closest_target.alpha
            self.car.steering_angle = (self.car.max_steering*2/np.pi)*np.arctan(alpha/dist**self.car.turning_sharpness)
            self.car.velocity.x = self.cruising_speed

        self.car.acceleration = max(-self.car.max_acceleration, min(self.car.acceleration, self.car.max_acceleration))
        self.car.steering_angle = max(-self.car.max_steering, min(self.car.steering_angle, self.car.max_steering))

    def midpoint_steering_angle(self):
        if (len(self.target.visible_targets) > 0 
        and np.linalg.norm(self.target.closest_target.position-self.car.position) < self.car.fov/self.ppu
        and np.linalg.norm(self.target.closest_target.position-self.car.position) > 20/self.ppu
        and self.car.auto == True 
        and self.target.closest_target.passed == False):
            
            dist = self.target.closest_target.dist_car
            #print("closest target dist : ", dist)
            #print()
            alpha = self.target.closest_target.alpha
            #print("closest target angle : ", alpha)
            midpoint_steering_angle = (self.car.max_steering*2/np.pi)*np.arctan(alpha/dist**self.car.turning_sharpness)
        else:
            midpoint_steering_angle = 0

        midpoint_steering_angle = max(-self.car.max_steering, min(midpoint_steering_angle, self.car.max_steering))

        return midpoint_steering_angle


    def midpoint_steering_angle_ros(self):
        if len(self.target.visible_targets) > 0:
            dist = self.target.closest_target.dist_car
            #print("closest target dist : ", dist)
            #print()
            alpha = self.target.closest_target.alpha
            #print("closest target angle : ", alpha)
            midpoint_steering_angle = (self.car.max_steering*2/np.pi)*np.arctan(alpha/dist**self.car.turning_sharpness)
        else:
            midpoint_steering_angle = 0

        midpoint_steering_angle = max(-self.car.max_steering, min(midpoint_steering_angle, self.car.max_steering))

        return midpoint_steering_angle

    def implement_main_logic(self, dt):
        self.car.update(dt)

        for target in self.target.targets:
            target.update(self)

        for category in Side:
            for cone in self.cone.cone_list[category]:
                cone.update(self)


    def get_observation_ros(self, cone_loc, num_obs, midpoint_obs = False):
        observation = np.zeros(num_obs)
        
        #resetting all objects
        self.cone = Cone(-1000,-1000,Side.LEFT) 
        self.car = Car(7,10)
        self.path = Path()
        self.target = Target(-1000,-1000)

        #assuming cone info is {'left' : [[r,theta]], 'right' : [[r,theta]]} where r is in cm, and angle is in degrees`
        cone_list = []
        for category in ['left', 'right']:
            for cone in cone_loc[category]:
                #filtering out cones and rescaling distance
                if cone[0] <= 1.25 and cone[1] > -120 and cone[1] < 120:
                    new_r = np.interp(cone[0], [0, 1.25], [0, self.car.fov/self.ppu])
                    cone = [new_r, cone[1], category]
                    cone_list.append(cone)

        new_cone_list_left = []
        new_cone_list_right = []
        #adding cone objects
        for cone in cone_list:
            r = cone[0]
            theta = cone[1]
            category = cone[2]
            x_pos = self.car.position.x + r * np.cos(radians(bound_angle_180(self.car.angle + theta)))
            y_pos = self.car.position.y + r * np.sin(-1 * radians(bound_angle_180(self.car.angle + theta)))

            if category == 'left':
                cone_to_init = Cone(x_pos, y_pos, 'left')
                cone_to_init.in_fov = True
                cone_to_init.visible = True
                new_cone_list_left.append(cone_to_init)

            elif category == 'right':
                cone_to_init = Cone(x_pos, y_pos, 'right')
                cone_to_init.in_fov = True
                cone_to_init.visible = True
                new_cone_list_right.append(cone_to_init)

        #update left cone lists
        self.cone.cone_list[Side.LEFT] =  new_cone_list_left
        self.cone.visible_cone_list[Side.LEFT] =  new_cone_list_left
        self.cone.in_fov_cone_list[Side.LEFT] =  new_cone_list_left

        #update right cone_lists
        self.cone.cone_list[Side.RIGHT] =  new_cone_list_right
        self.cone.visible_cone_list[Side.RIGHT] =  new_cone_list_right
        self.cone.in_fov_cone_list[Side.RIGHT] =  new_cone_list_right

        #for category in Side:
            #for cone in self.cone.cone_list[category]:
                #print()
                #print("Cone x pos : ", cone.position.x)
                #print("Cone y pos : ", cone.position.y)
                #print()


        #setting booleans to true in order to trigger updates
        for category in Side:
            self.cone.new_in_fov_cone_flag[category] = True
            self.cone.new_visible_cone_flag[category] = True

        #update cones and boundary sample
        for category in Side:
            for cone in self.cone.cone_list[category]:
                cone.update_ros(self)

        #calculate targets
        self.path.generate_midpoint_path(self)

        #update targets
        for target in self.target.targets:
            target.update(self)
            target.visible = True

        self.target.update_target_lists()
        self.target.update_closest_target()

        #update boundary
        self.cone.update_cone_list(self)

        #print()
        #print("cone list left length : ", len(self.cone.in_fov_cone_list[Side.LEFT]))
        #print()
        #print("cone list right length : ", len(self.cone.in_fov_cone_list[Side.RIGHT]))
        #print()
        #print("target list length : ", len(self.target.targets))
        #print()
        #print("target locations : ", self.target.target_locations)
        #print()
        #print("boundary points left : ", self.cone.polar_boundary_sample[Side.LEFT])
        #print()
        #print("boundary points right : ", self.cone.polar_boundary_sample[Side.RIGHT])

        
        midpoint_steering_angle = self.midpoint_steering_angle_ros()
        print()
        print("midpoint steering angle", midpoint_steering_angle)
    
        #create observation
        if midpoint_obs:
            observation[0] = np.interp(midpoint_steering_angle, [-120,120], [-1, 1])

        for i, cone in enumerate(self.cone.polar_boundary_sample[Side.LEFT]):
            observation[0 + 2*i + int(midpoint_obs)] = np.interp(cone[0], [0, self.car.fov/self.ppu], [-1, 1])
            observation[1 + 2*i + int(midpoint_obs)] = np.interp(cone[1], [-120, 120], [-1, 1])

        for i, cone in enumerate(self.cone.polar_boundary_sample[Side.RIGHT]):
            observation[10 + 2*i + int(midpoint_obs)] = np.interp(cone[0], [0, self.car.fov/self.ppu], [-1, 1])
            observation[11 + 2*i + int(midpoint_obs)] = np.interp(cone[1], [-120, 120], [-1, 1])


        spline_error = False
        for i, cone in enumerate(self.cone.polar_boundary_sample[Side.LEFT]):
            if cone[1] > 120:
                spline_error = True
                print("SplineError - boundary spline computed incorrectly : ", cone[1])

        for i, cone in enumerate(self.cone.polar_boundary_sample[Side.RIGHT]):
            if cone[1] > 120:
                spline_error = True
                print("SplineError - boundary spline computed incorrectly : ", cone[1])
        
        #print(observation)

        return observation, spline_error, midpoint_steering_angle


    def set_done(self, episode_time_running, episode_num, num_steps):
        self.path.compute_boundaries(self)
        self.car.car_crash_mechanic(self.cone, self.path)
        episode_ending = None
            
        if self.car.crashed:
            self.done = True
            print('car crashed : ' + self.LEVEL_ID)
            episode_ending = ('crash', self.LEVEL_ID, episode_num, num_steps)
            return True, episode_ending

        elif self.track_number == 2:
            print("track complete! : " + self.LEVEL_ID)
            episode_ending = ('success', self.LEVEL_ID, episode_num, num_steps)
            self.done = True
            return True, episode_ending

        elif int(episode_time_running) > 200:
            print('timelimit reached : ' + self.LEVEL_ID)
            self.done = True
            episode_ending = ('time limit', self.LEVEL_ID, episode_num, num_steps)
            return True, episode_ending

        else:    
            self.done = False
            return False, episode_ending

    def load_observation_statistics(self, path):
        self.observation_statistics = np.genfromtxt(path, delimiter=',')

    def get_observation(self, num_obs, midpoint_obs, steering_angle_moving_average, random_guide, robustness_test, robustness_scale):

        observation = np.zeros(num_obs)
        midpoint_steering_angle = self.midpoint_steering_angle()

        #if steering_angle_moving_average != None:
        #    steering_angle_average = exponential_weighted_average(steering_angle_moving_average)

        #observation[0] = np.interp(self.car.velocity.x, [0, self.car.max_velocity], [-1, 1])
        #observation[1] = np.interp(self.car.angle, [-180,180], [-1, 1])

        if midpoint_obs:
            observation[0] = np.interp(midpoint_steering_angle, [-120,120], [-1, 1])

        if random_guide:
            observation[0] = np.random.normal(loc = 0.0, scale = 0.5)
            observation[0] = np.clip(observation[0], -1, 1)
        
        #observation[2] = np.interp(steering_angle_average, [-120,120], [-1, 1])

        for i, cone in enumerate(self.cone.polar_boundary_sample[Side.LEFT]):
            observation[0 + 2*i + int(midpoint_obs)] = np.interp(cone[0], [0, self.car.fov/self.ppu], [-1, 1])
            observation[1 + 2*i + int(midpoint_obs)] = np.interp(cone[1], [-120, 120], [-1, 1])
            #observation[2 + 2*i] = np.interp(np.clip((np.interp(cone[0], [0, self.car.fov/self.ppu], [-1, 1]) - self.observation_statistics[2 + 2*i, 0])/self.observation_statistics[2 + 2*i, 1], -5,5), [-5, 5], [-1, 1])
            #observation[3 + 2*i] = np.interp(np.clip((np.interp(cone[1], [-180, 180], [-1, 1]) - self.observation_statistics[3 + 2*i, 0])/self.observation_statistics[3 + 2*i, 1], -5,5), [-5, 5], [-1, 1])

        for i, cone in enumerate(self.cone.polar_boundary_sample[Side.RIGHT]):
            observation[10 + 2*i + int(midpoint_obs)] = np.interp(cone[0], [0, self.car.fov/self.ppu], [-1, 1])
            observation[11 + 2*i + int(midpoint_obs)] = np.interp(cone[1], [-120, 120], [-1, 1])
            #observation[12 + 2*i] = np.interp(np.clip((np.interp(cone[0], [0, self.car.fov/self.ppu], [-1, 1]) - self.observation_statistics[12 + 2*i, 0])/self.observation_statistics[12 + 2*i, 1], -5,5), [-5, 5], [-1, 1])
            #observation[13 + 2*i] = np.interp(np.clip((np.interp(cone[1], [-180, 180], [-1, 1]) - self.observation_statistics[13 + 2*i, 0])/self.observation_statistics[13 + 2*i, 1], -5,5), [-5, 5], [-1, 1])

        if robustness_test:
            for i in range(len(observation)):
                observation[i] = observation[i] + np.random.normal(loc = 0.0, scale = robustness_scale)
                observation[i] = np.clip(observation[i], -1, 1) 

        return observation

    def run(self, method = "autonomous"):

        self.initialize_images()

        if method == "autonomous":
            self.initialize_map()
        else:
            self.car.auto = False
        
        time_start = time.time()

        while not self.exit and not self.done:
            
            self.num_steps += 1

            dt = self.clock.get_time() / 500

            # Event queue
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.exit = True

            if method == "autonomous":
                pp_functions.manual_controls.enable_dragging_screen(self, events)
            else:
                # user inputs
                pp_functions.manual_controls.user_input(self, events, dt)
                         
            # Defining the time running since simulation started
            self.time_running = time.time() - time_start
            
            #redefining the car angle so that it is in (-180,180)
            self.car.config_angle()

            #update target list
            self.target.update_target_lists()
           
            #update cone list
            self.cone.update_cone_list(self)
            
            #calculate closest target
            self.target.update_closest_target()
                
            #reset targets for new lap
            self.reset_new_lap()
            
            #automatic steering
            self.steering()
            
            #computing boundary estimation
            self.path.compute_boundaries(self)

            #compute midpoint path
            self.path.generate_midpoint_path(self)
                   
            #implement track logic
            self.track_logic()

            #car crash logic 
            self.car.car_crash_mechanic(self.cone, self.path)

            #Calculate reward
            #self.reward = calculate_reward(self)a

            #checking exit conditions
            #self.set_done(self.time_running, self.episode_num, self.num_steps)
                    
            # Logic
            self.implement_main_logic(dt)
            
            #Drawing
            pp_functions.drawing.render(self)

            self.clock.tick(self.ticks)

        pygame.quit()
        

if __name__ == '__main__':
    sim = PathPlanning()

    # 2 methods:
    #   1) autonomous: no user inputs, only screen dragging
    #   2) manual: old simulation with user inputs

    sim.run(method = "manual") 