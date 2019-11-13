# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import pandas as pd

def fmod_pos(x,y):
    temp = np.fmod(x,y)
    if temp < 0.0:
        temp = temp + y
    return temp

def fmod_pos_array(x,y):
    temp = np.zeros(len(x))
    for i,xi in enumerate(x):
        temp[i] = fmod_pos(xi,y)
    return temp

def section(k_1,d_1,k_2,d_2): # intersection of two lines
    section_x = (d_1-d_2)/(k_2-k_1)
    section_y = k_1*section_x+d_1
    return (section_x,section_y)


class ModuleIllumination:
    def __init__(self, module_length=1.92, module_tilt=52, mount_height=0.5,
                 module_distance=7.1, dni=1, dhi=1, zenith_sun=30, azimuth_sun=150,
                 albedo=0.3, ground_steps=101, module_steps=12, angle_steps=180):
        '''
        Simulation of illumination for a bifacial solar panel in a periodic
        south facing array.

        Accepts array inputs for sun position for fast tiem series evaluation.

        Parameters
        ----------
        module_length : numeric
            length of the module used for the array

        module_tilt : numeric
            tilt angle of the module

        mount_height : numeric
            Mounting height of the modules. Difined as distance between the lowest
            point of the module and the ground

        module_distance:
            Distance between modules. Defined as distance between lowest point
            of one module to the same point on the next row.

        dni : numeric
            Direct normal irradiance

        dhi : numeric
            Diffuse horizontal irradiance

        zenith_sun : numeric or array-like
            Zenith angle of the sun in degrees

        azimuth_sun : numeric or array-like
            Azimuth angle of the sun in degrees. 180 degrees defined as south.

        albedo : numeric
            Albedo of the ground. Should have value between 0 and 1.

        ground_steps : int
            Resolution on the ground where the irradiance is evaluated

        module_steps : int
            Resolution on the module where the irradiance is evaluated

        angle_steps : int
            Angular resolution of the ground radiance
        '''
        self.L = module_length
        self.theta_m_rad = module_tilt
        self.H = mount_height
        self.dist= module_distance
        self.DNI = dni
        self.DHI = dhi
        self.theta_S_rad = zenith_sun
        self.phi_S_rad = azimuth_sun
        self.albedo = albedo
        self.ground_steps = ground_steps
        self.module_steps = module_steps
        self.angle_steps  = angle_steps

        #Define variables derived from these base variables
        self.x_g_array = np.linspace(0,self.dist,self.ground_steps)
        self.x_g_distance = self.dist/(self.ground_steps-1) # distance between two points on x_g_array
        #self.l_array = np.linspace(0,self.L,self.module_steps) # OLD, changed on 26 April 2019!
        self.l_array = np.linspace(self.L/self.module_steps,self.L,self.module_steps)-0.5*self.L/self.module_steps
        # normal angle of the sun
        self.n_S = np.array([np.sin(self.theta_S_rad)*np.cos(-self.phi_S_rad),
                             np.sin(self.theta_S_rad)*np.sin(-self.phi_S_rad),
                             np.cos(self.theta_S_rad)])
        # initializing the results dictionary
        self.results = {}

        self.module()
        self.calc_irradiance_module_sky_direct()
        self.calc_irradiance_module_sky_diffuse()
        self.calc_radiance_ground_direct()
        self.calc_radiance_ground_diffuse()
        self.calc_module_ground_matrix()
        self.calc_irradiance_module_ground_direct()
        self.calc_irradiance_module_ground_diffuse()


    def module(self): # some functions and values for the PV module
        '''
        Helper function to introduce some characteristic points and vectors of the module
        '''
        self.H_m = self.L*np.sin(self.theta_m_rad)
        self.e_m = np.array([np.cos(self.theta_m_rad),np.sin(self.theta_m_rad)]) # unit vector along the module
        self.n_m = np.array([-np.sin(self.theta_m_rad),np.cos(self.theta_m_rad)]) # normal to the module
        self.n_m_3D = np.array([self.n_m[0],0,self.n_m[1]]) # normal to the module

    # IRRADIANCE ON MODULE FROM THE SKY
    def calc_irradiance_module_sky_direct(self):
        '''
        Calculates the direct irradiance on the module for one or a series of
        solar positions.
        '''

        try:
            temp_irrad = np.zeros((self.n_S.shape[1], self.module_steps))
        except:
            temp_irrad = np.zeros(self.module_steps)

        self.cos_alpha_mS = np.dot(self.n_S.T, self.n_m_3D) # cosine of angle between Sun and module normal
        angle_term = np.cos(self.theta_m_rad)-np.sin(self.theta_m_rad)*self.n_S[0]/self.n_S[2] # needed for calculating shadow on module

        l_shadow = np.where(self.cos_alpha_mS > 0,
                            self.L-self.dist/angle_term,
                            self.L+self.dist/angle_term)

        try:
            temp_irrad[:] = self.DNI*self.cos_alpha_mS[:,None]
            temp_irrad[np.greater.outer(l_shadow, self.l_array)] = 0
            temp_front = np.where((self.cos_alpha_mS > 0)[:,None], temp_irrad, 0)
            temp_back = np.where((self.cos_alpha_mS < 0)[:,None], -temp_irrad, 0)
        except:
            temp_irrad[:] = self.DNI*self.cos_alpha_mS
            temp_irrad[np.greater.outer(l_shadow, self.l_array)] = 0
            temp_front = np.where((self.cos_alpha_mS > 0), temp_irrad, 0)
            temp_back = np.where((self.cos_alpha_mS < 0), -temp_irrad, 0)

        self.results['irradiance_module_front_sky_direct'] = temp_front
        self.results['irradiance_module_back_sky_direct']  = temp_back
        self.results['irradiance_module_front_sky_direct_mean'] = np.mean(temp_front, axis=-1)
        self.results['irradiance_module_back_sky_direct_mean']  = np.mean(temp_back, axis=-1)

    def calc_irradiance_module_sky_diffuse(self):
        '''
        Calculates the irradiance of diffuse sky on the module front.
        The result is only depended on the geometrie of the solar panel array.
        '''

        vectors_front = np.multiply.outer(self.L-self.l_array, self.e_m) -\
                  np.array([self.dist,0])

        cos_alpha_2 = (np.dot(vectors_front, self.n_m)/\
                      np.linalg.norm(vectors_front, axis=1))
        sin_alpha_2 = (1-cos_alpha_2**2)**0.5
        irradiance_front = (sin_alpha_2+1)/2.0

        vectors_back = np.multiply.outer(self.L-self.l_array, self.e_m) +\
                             np.array([self.dist,0])
        cos_epsilon_1 = np.dot(vectors_back, -self.n_m)/norm(vectors_back, axis=1)
        sin_epsilon_2 = (1-cos_epsilon_1**2)**0.5
        irradiance_back = (1-sin_epsilon_2)/2

        self.results['irradiance_module_front_sky_diffuse'] = irradiance_front
        self.results['irradiance_module_front_sky_diffuse_mean'] = irradiance_front.mean()

        self.results['irradiance_module_back_sky_diffuse'] = irradiance_back
        self.results['irradiance_module_back_sky_diffuse_mean'] = irradiance_back.mean()

    def calc_shadow_field(self):
        '''
        Calculates the start and end position of shadow of direct sunlight.
        Depends on sun position and array  geometry.

        Returns
        -------
        shadow_start : start position of shadow relativ to (0, 0)
        shadow_end : end position of shadow relativ to (0, 0)
        '''

        #calculating shadow position for B0
        shadow_B = -self.H / self.n_S[2] * self.n_S[0]

        #calculating shadow for D0
        D0 = self.L*self.e_m + np.array([0, self.H])
        shadow_D = -D0[1] / self.n_S[2] * self.n_S[0] + D0[0]

        #if shadow position of D0 is smaller then B0, positions need to be flipped.
        flipp_mask = shadow_D < shadow_B
        shadow_start, shadow_end = np.where(flipp_mask, shadow_D, shadow_B),\
                   np.where(flipp_mask, shadow_B, shadow_D)
        return shadow_start, shadow_end

    def calc_radiance_ground_direct(self):
        '''
        Calculates the position resolved direct irradiance on the ground.
        '''
        shadow_start, shadow_end = self.calc_shadow_field()
        length_shadow = shadow_end - shadow_start

        # reduce such that start and end is in [0, dist] unit cell
        shadow_start_uc = np.remainder(shadow_start, self.dist)
        shadow_end_uc = np.remainder(shadow_end, self.dist)

        #if length_shadow < self.dist: # only in this case direct Sunlight will hit the ground
        length_shadow = shadow_end - shadow_start
        shadow_filter = length_shadow < self.dist
        shadow_start_uc = np.where(shadow_filter, shadow_start_uc, 0)
        shadow_end_uc = np.where(shadow_filter, shadow_end_uc, self.dist)

        #if the ground position is smaller then shadow start OR larger then
        # shadow end it is directly illuminated if shadow start (uc) < shadow end (uc)
        illum_array_1 = (np.greater.outer(shadow_start_uc, self.x_g_array,) | \
                         np.less.outer(shadow_end_uc, self.x_g_array))

        #if the ground position is smaller then shadow start AND larger then
        # shadow end it is directly illuminated if shadow start (uc) > shadow end (uc)
        illum_array_2 = (np.greater.outer(shadow_start_uc, self.x_g_array,) & \
                         np.less.outer(shadow_end_uc, self.x_g_array))

        #choose appropriet illumination array

        try:
            illum_array_temp = np.where((shadow_end_uc >= shadow_start_uc)[:,None],
                                    illum_array_1,
                                    illum_array_2)
            illum_array_temp = ((illum_array_temp*self.DNI)*np.cos(self.theta_S_rad)[:, None])
        except:
            illum_array_temp = np.where((shadow_end_uc >= shadow_start_uc),
                                    illum_array_1,
                                    illum_array_2)
            illum_array_temp = ((illum_array_temp*self.DNI)*np.cos(self.theta_S_rad))

        self.results['radiance_ground_direct_emitted'] = illum_array_temp / np.pi * self.albedo


    def calc_sin_B_i(self, i, x_g):
        '''
        Calculates cosinus between array of ground positions x_g and point B_i
        with respect to the surface normal
        (lower end of i'th module)

        Parameters
        ----------
        i : int
            Index of corresponding module
        x_g : numpy array
            array of ground positions

        Returns
        -------
        cosinus : array of cosinus for every position in ground position array
        '''
        x = np.subtract.outer(i*self.dist, x_g)
        y = self.H
        return x/norm([x, y])

    def calc_sin_D_i(self,i,x_g):
        '''
        Calculates cosinus between array of ground positions x_g and point D_i
        with respect to the surface normal
        (lower end of i'th module)

        Parameters
        ----------
        i : int
            Index of corresponding module
        x_g : numpy array
            array of ground positions

        Returns
        -------
        cosinus : array of cosinus for every position in ground position array
        '''
        D_0 = self.L*self.e_m
        x = np.subtract.outer(i*self.dist, x_g)+D_0[0]
        y = self.H +D_0[1]
        return x/norm([x, y])

    # radiance of the ground originating from diffuse skylight
    # this method leads to the same results as the one developped earlier (v0.1) but is much faster.
    def calc_radiance_ground_diffuse(self):
        '''
        Calculates the position resolved diffuse irradiance on the ground.
        '''

        #Check how many periods have to be considered in negativ direction
        lower_bound = 0
        while True:
           cos_B_x = self.calc_sin_B_i(lower_bound, self.x_g_array)
           cos_D_x = self.calc_sin_D_i(lower_bound-1, self.x_g_array)

           #check if sky is visible for any point between B_x and D_x-1
           if all(cos_B_x<cos_D_x):
               break
           lower_bound = lower_bound - 1

        #Check how many periods have to be considered in positive direction
        upper_bond = 0
        while True:
           B0 = self.calc_sin_B_i(upper_bond, self.x_g_array)
           B1 = self.calc_sin_B_i(upper_bond+1, self.x_g_array)
           D0 = self.calc_sin_D_i(upper_bond, self.x_g_array)
           D1 = self.calc_sin_D_i(upper_bond+1, self.x_g_array)
           #check if sky is visible between module x and x+1
           if all(np.maximum(B0, D0)>np.minimum(B1, D1)):
               break
           upper_bond = upper_bond + 1


        sin_eta_arr  = self.calc_sin_B_i(np.arange(lower_bound, upper_bond+1), self.x_g_array)
        sin_zeta_arr = self.calc_sin_D_i(np.arange(lower_bound, upper_bond+1), self.x_g_array)

        arr_eta_zeta = np.stack([sin_eta_arr, sin_zeta_arr])
        #sort such that smallest sin is always first in array
        arr_eta_zeta.sort(axis=0)

        #substract lower angle of ith row from higher angle of i+1 'th row
        sky_view_factors = np.roll(arr_eta_zeta[0], -1, axis=0) - arr_eta_zeta[1]
        #set negative sky_view_factors to zero
        sky_view_factors[sky_view_factors<0] = 0
        #sum over all "windows" between module rows
        illum_array = sky_view_factors.sum(axis=0)/2

        irradiance_ground_diffuse_received = illum_array * self.DHI

        # division by pi converts irradiance into radiance assuming Lambertian scattering
        self.results['radiance_ground_diffuse_emitted']  = irradiance_ground_diffuse_received / np.pi * self.albedo


    # IRRADIANCE ON MODULE FROM THE GROUND
    # some functions
    def alpha(self,vector,front_back): #calculate angle between n_m and vector for front side
        if front_back == 'front':
            cos_alpha = np.dot(vector, self.n_m)/np.linalg.norm(vector)
        elif front_back == 'back':
            cos_alpha = np.dot(vector,-self.n_m)/np.linalg.norm(vector)
        else:
            print('ERROR! Value neither front nor back')
            cos_alpha = 1./0.
        sign_alpha = np.sign(self.n_m[0]*vector[1]-self.n_m[1]*vector[0])#sign such that vectors pointing below n_m positive
        return sign_alpha*np.arccos(cos_alpha)


    #calculate matrix that determines distribution of light from ground on the module
    def calc_module_ground_matrix(self):
        for fb in ['front','back']:
            intensity_matrix = np.zeros((self.module_steps, self.angle_steps, self.ground_steps))
            field_name = 'module_' + fb + '_ground_matrix'
            t = time.process_time()
            for i,l in enumerate(self.l_array): # remove 0th entry as otherwise the ray to calculate alpha1 would be parallel to x-axis!!!
                vector_2 = -l*self.e_m
                # position of the line starting at point on module via lowest pt. of module
                x_g_2 = l*self.e_m[0] - (l*self.e_m[1]+self.H)/vector_2[1]*vector_2[0]
                if fb == 'front':
                    vector_1 = np.array([-self.dist,0])-l*self.e_m
                if fb == 'back':
                    vector_1 = np.array([ self.dist,0])-l*self.e_m
                x_g_1 = l*self.e_m[0] - (l*self.e_m[1]+self.H)/vector_1[1]*vector_1[0]
                if fb == 'front':
                    lower_index = int(round(x_g_1/self.x_g_distance))
                    upper_index = int(round(x_g_2/self.x_g_distance))
                if fb == 'back':
                    lower_index = int(round(x_g_2/self.x_g_distance))
                    upper_index = int(round(x_g_1/self.x_g_distance))
                index_array = np.arange(lower_index, upper_index+1)
                x_g_array_full = self.x_g_distance*index_array
                alpha_array = np.zeros(len(index_array))
                #addend_array = np.zeros(len(index_array))
                for j,x_g in enumerate(x_g_array_full):
                    vector = np.array([x_g,-self.H])-l*self.e_m
                    alpha_array[j] = self.alpha(vector,fb)
                self.test = 0
                self.test2 = np.sin(alpha_array[-1]) - np.sin(alpha_array[0])
                self.upper = alpha_array[-1]
                self.lower = alpha_array[0]
                for j,x_g in enumerate(x_g_array_full):
                    if j == 0:
                        delta_alpha = 0.5*(alpha_array[1]-alpha_array[0])
                    elif j == len(index_array)-1:
                        delta_alpha = 0.5*(alpha_array[j]-alpha_array[j-1])
                    else:
                        delta_alpha = 0.5*(alpha_array[j+1]-alpha_array[j-1])
                    #print(np.mod(index_array[j],self.ground_steps))
                    cos_alpha = abs(np.cos(alpha_array[j]))
                    angle_index = np.int(np.floor(alpha_array[j]*self.angle_steps/np.pi+self.angle_steps/2.0))
                    if angle_index == self.angle_steps:
                        angle_index = angle_index-1
                    ground_index = np.mod(index_array[j],self.ground_steps)
                    if angle_index >= self.angle_steps:
                        angle_index = self.angle_steps -1
                        print('Somethings fishy here')

                    self.test += abs(cos_alpha)*abs(delta_alpha)
                    intensity_matrix[i,angle_index,ground_index] += np.pi/2.0*cos_alpha*abs(delta_alpha)
            self.results[field_name] = intensity_matrix
            self.results[field_name + '_time'] = time.process_time()-t

    #irradiance on the module from the ground originating from direct skylight
    def calc_irradiance_module_ground_direct(self):
        for fb in ['front','back']:
            field_name = 'irradiance_module_' + fb + '_ground_direct'
            matrix = self.results['module_' + fb + '_ground_matrix']
            temp = np.sum(matrix, axis = 1)
            self.results[field_name] = self.results['radiance_ground_direct_emitted']@temp.T
            self.results[field_name + '_mean'] = np.mean(self.results[field_name], axis=-1)

    #irradiance on the module from the ground originating from diffuse skylight
    def calc_irradiance_module_ground_diffuse(self):
        for fb in ['front','back']:
            field_name = 'irradiance_module_' + fb + '_ground_diffuse'
            matrix = self.results['module_' + fb + '_ground_matrix']
            temp = np.sum(matrix, axis = 1)
            self.results[field_name] = (temp*self.results['radiance_ground_diffuse_emitted']).sum(axis=1)
            self.results[field_name + '_mean'] = np.mean(self.results[field_name])
