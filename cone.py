from math import sin, radians, degrees, copysign
from pygame.math import Vector2
import numpy as np
from enum import Enum
from scipy.interpolate import splprep, splev
from operator import itemgetter

class Side(Enum):
    LEFT = 1
    RIGHT = 2

class Cone:
    def __init__(self, x, y, category):
        self.position = Vector2(x, y)
        self.image = {Side.LEFT: None, Side.RIGHT: None}    
        self.visible = False   
        self.in_fov = False
        self.category = category
        self.dist_car = 10**10
        self.alpha = 0
        

        self.cone_list = {Side.LEFT:[], Side.RIGHT: []}
        self.polar_cone_list = []
        self.visible_cone_list = {Side.LEFT:[], Side.RIGHT: []}
        self.in_fov_cone_list = {Side.LEFT:[], Side.RIGHT: []}

        self.id = np.random.randint(1, 10**10)
        self.id_list = {Side.LEFT:[], Side.RIGHT: []} 
        self.in_fov_id_list = {Side.LEFT:[], Side.RIGHT: []} 

        self.new_visible_cone_flag = {Side.LEFT: False, Side.RIGHT: False}
        self.new_in_fov_cone_flag = {Side.LEFT: False, Side.RIGHT: False}
        
        self.first_cone_found = {Side.LEFT: False, Side.RIGHT: False}
        self.first_visible_cone = {Side.LEFT: 0, Side.RIGHT: 0}

        self.boundary_sample = {Side.LEFT:[[-100 for _ in range(5)], [-100 for _ in range(5)]], Side.RIGHT: [[-100 for _ in range(5)], [-100 for _ in range(5)]]} # [[x_pos_list], [y_pos_list]]
        self.boundary_sample_draw = {Side.LEFT:[[-100 for _ in range(30)], [-100 for _ in range(30)]], Side.RIGHT: [[-100 for _ in range(30)], [-100 for _ in range(30)]]} # [[x_pos_list], [y_pos_list]]
        self.polar_boundary_sample = {Side.LEFT:[], Side.RIGHT: []} #[[r1, theta1], [r2, theta2], ...]

    def update_cone_list(self, pp): 
        
        self.polar_cone_list = []       
        for category in Side:

            initial_length_visible = len(self.visible_cone_list[category]) 
            initial_length_in_fov = len(self.in_fov_cone_list[category])
            initial_in_fov_id_list = self.in_fov_id_list[category]
            
            self.visible_cone_list[category] = []
            self.in_fov_cone_list[category] = []
            self.in_fov_id_list[category] = []

            for cone in self.cone_list[category]:
                self.id_list[category].append(cone.id)
                self.polar_cone_list.append([cone.alpha, cone.dist_car, cone.category])
                if cone.visible == True:
                    self.visible_cone_list[category].append(cone)

                if cone.in_fov == True:
                    self.in_fov_cone_list[category].append(cone)
                    self.in_fov_id_list[category].append(cone.id)

            if initial_length_visible != len(self.visible_cone_list[category]):
                self.new_visible_cone_flag[category] = True
            else:
                self.new_visible_cone_flag[category] = False

            for id in self.in_fov_id_list[category]:
                if id not in initial_in_fov_id_list:
                    self.new_in_fov_cone_flag[category] = True
                    break
                else:
                    self.new_in_fov_cone_flag[category] = False

      #update boundary sample and polar boundary sample
        if pp.car.auto:
            num_samples = 5
            for category in Side:
                cone_list_x_pos = []
                cone_list_y_pos = []
                cone_dists = []
                cone_list_x_pos_sorted = []
                cone_list_y_pos_sorted = []
                list_to_sort = []

                if (self.new_in_fov_cone_flag[Side.LEFT] == True or self.new_in_fov_cone_flag[Side.RIGHT] == True):
                   #(self.new_visible_cone_flag[Side.LEFT] == True or self.new_visible_cone_flag[Side.RIGHT] == True): 
                    for cone in self.in_fov_cone_list[category]:
                        cone_list_x_pos.append(cone.position.x)
                        cone_list_y_pos.append(cone.position.y)
                        cone_dists.append(cone.dist_car)

                    cone_pos = [cone_list_x_pos, cone_list_y_pos]
                 
			        #couple each cone with its distance to the car	 
                    for i in range(len(cone_pos[0])):
                        list_to_sort.append([cone_dists[i], cone_pos[0][i], cone_pos[1][i]])

			        #ordering the cones by distance from car   
                    list_to_sort = sorted(list_to_sort, key=itemgetter(0))

                    if len(list_to_sort) > 1:
                        for i in range(len(list_to_sort)):
					        #if statement making sure we have no duplicate co-ordinates
                            if list_to_sort[i][1] in cone_list_x_pos_sorted or list_to_sort[i][2] in cone_list_y_pos_sorted:
                                pass
                            else:
                                cone_list_x_pos_sorted.append(list_to_sort[i][1])
                                cone_list_y_pos_sorted.append(list_to_sort[i][2])
                    
                    cone_pos_sorted = [cone_list_x_pos_sorted, cone_list_y_pos_sorted]

                    if len(cone_pos_sorted[0]) <= 1:
                        spline_points = [[0 for _ in range(5)],[0 for _ in range(5)]]
                        spline_points_draw = [[0 for _ in range(30)],[0 for _ in range(30)]]
                        self.boundary_sample[category] = spline_points
                        self.boundary_sample_draw[category] = spline_points_draw

                    elif len(cone_pos_sorted[0]) == 2:
                        tck, u = splprep(cone_pos_sorted, s=0.1, k = 1)

                        start = 0
                        stop = 1.01

                        unew = np.arange(start, stop, 1/(num_samples - 1)) 
                        spline_points = splev(unew, tck)
                        self.boundary_sample[category] = spline_points

                        u_draw = np.arange(start, stop, 1/30) 
                        spline_points_draw = splev(u_draw, tck)
                        self.boundary_sample_draw[category] = spline_points_draw
                    
                    elif len(cone_pos_sorted[0]) > 2:
                        tck, u = splprep(cone_pos_sorted, s=0.1, k = 2) 

                        start = 0
                        stop = 1.01

                        unew = np.arange(start, stop, 1/(num_samples - 1)) 
                        spline_points = splev(unew, tck)
                        self.boundary_sample[category] = spline_points

                        u_draw = np.arange(start, stop, 1/30) 
                        spline_points_draw = splev(u_draw, tck)
                        self.boundary_sample_draw[category] = spline_points_draw

                    
            
                self.polar_boundary_sample[category] = []

                #updating polar_boundary_sample
                for i in range(num_samples):
                    dist = np.linalg.norm(Vector2(self.boundary_sample[category][0][i], self.boundary_sample[category][1][i]) - pp.car.position)
                    #print('dist: ', dist)
                    #calculating angle between car angle and sample point (alpha)
                    a_b = Vector2(self.boundary_sample[category][0][i], self.boundary_sample[category][1][i]) - pp.car.position
                    a_b = np.transpose(np.matrix([a_b.x,-1*a_b.y ]))
                    rotate = np.matrix([[np.cos(-pp.car.angle*np.pi/180),-1*np.sin(-pp.car.angle*np.pi/180)],
                                        [np.sin(-pp.car.angle*np.pi/180),np.cos(-pp.car.angle*np.pi/180)]])
                    a_b = rotate*a_b
                    a = a_b[0] + np.random.normal(loc = 0, scale = 1e-8)
                    b = a_b[1] + np.random.normal(loc = 0, scale = 1e-8)
                    beta = np.arctan(b/a)*(180/np.pi)
                    alpha = beta + 90*(b/np.abs(b))*np.abs((a/np.abs(a)) - 1)
                    angle = alpha[0,0]

                    self.polar_boundary_sample[category].append([dist, angle])
        
    def update(self, pp): 
        
        #distance to car
        self.dist_car = np.linalg.norm(self.position - pp.car.position)
        
        #calculating angle between car angle and cone (alpha)
        a_b = self.position - pp.car.position
        a_b = np.transpose(np.matrix([a_b.x,-1*a_b.y ]))
        rotate = np.matrix([[np.cos(-pp.car.angle*np.pi/180),-1*np.sin(-pp.car.angle*np.pi/180)],
                            [np.sin(-pp.car.angle*np.pi/180),np.cos(-pp.car.angle*np.pi/180)]])
        a_b = rotate*a_b
        a = a_b[0] + np.random.normal(loc = 0, scale = 1e-8)
        b = a_b[1] + np.random.normal(loc = 0, scale = 1e-8)
        beta = np.arctan(b/a)*(180/np.pi)
        alpha = beta + 90*(b/np.abs(b))*np.abs((a/np.abs(a)) - 1)
        self.alpha = alpha[0,0]

        #if cone within car fov, set to visible
        if self.dist_car < pp.car.fov / pp.ppu and np.abs(self.alpha) < pp.car.fov_range:
            self.visible = True
            self.in_fov = True
        else:
            self.in_fov = False


    def update_ros(self, pp): 
        
        #distance to car
        self.dist_car = np.linalg.norm(self.position - pp.car.position)
        
        #calculating angle between car angle and cone (alpha)
        a_b = self.position - pp.car.position
        a_b = np.transpose(np.matrix([a_b.x,-1*a_b.y ]))
        rotate = np.matrix([[np.cos(-pp.car.angle*np.pi/180),-1*np.sin(-pp.car.angle*np.pi/180)],
                            [np.sin(-pp.car.angle*np.pi/180),np.cos(-pp.car.angle*np.pi/180)]])
        a_b = rotate*a_b
        a = a_b[0] + np.random.normal(loc = 0, scale = 1e-8)
        b = a_b[1] + np.random.normal(loc = 0, scale = 1e-8)
        beta = np.arctan(b/a)*(180/np.pi)
        alpha = beta + 90*(b/np.abs(b))*np.abs((a/np.abs(a)) - 1)
        self.alpha = alpha[0,0]