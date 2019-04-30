# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.integrate import quad
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

def deg2rad(theta_deg):
    return theta_deg*np.pi/180.

def rad2deg(theta_rad):
    return theta_rad/np.pi*180.

def calc_irradiance_ground_direct(self):
        illum_array_temp = np.zeros(self.ground_steps)
        # x-coordinates of shadow:

        s_A = -self.H/self.n_S[2]
        x_A = s_A * self.n_S[0]
        s_B = (-self.H-self.L*np.sin(self.theta_m_rad))/self.n_S[2]
        x_B = self.L*np.cos(self.theta_m_rad)+s_B * self.n_S[0]

        flipp_mask = x_B < x_A
        x_A, x_B = np.where(flipp_mask, x_B, x_A),\
                   np.where(flipp_mask, x_A, x_B)

        length_shadow = x_B - x_A
        if length_shadow < self.D: # only in this case direct Sunlight will hit the ground
            x_A_red = fmod_pos(x_A,self.D) # reduce such that x_A_red is in [0,D]
            # x_B_red = x_B - x_A + x_A_red
            # calculate no of elements that are in the shadow
            illum_array_temp = self.x_g_array > length_shadow
            # roll the beginning of the shadowed area to x_A_red
            illum_array_temp = np.roll(illum_array_temp,
                                       np.round(x_A_red/3*self.ground_steps).astype(int))

            self.irradiance_ground_direct_received = illum_array_temp * self.DNI * np.cos(self.theta_S_rad)
            self.radiance_ground_direct_emitted  = self.irradiance_ground_direct_received / np.pi * self.albedo # division by pi converts irradiance into radiance assuming Lambertian scattering
        else:
            self.irradiance_ground_direct_received = np.zeros(self.ground_steps)
            self.radiance_ground_direct_emitted = np.zeros(self.ground_steps)

class ModuleIllumination:
    def __init__(self, InputDict):
        self.input = InputDict
# =============================================================================
#         self.variables()
#         self.module()
#         self.calc_irradiance_module_sky_direct()
#         self.calc_irradiance_module_sky_diffuse()
#         self.calc_radiance_ground_direct()
#         self.calc_radiance_ground_diffuse()
#         self.calc_module_ground_matrix()
#         self.calc_irradiance_module_ground_direct()
#         self.calc_irradiance_module_ground_diffuse()
# =============================================================================

    def variables(self): # Import all the variables
        self.L = self.input['L']
        self.theta_m_rad = deg2rad(self.input['theta_m_deg'])
        self.H = self.input['H']
        self.D = self.input['D']
        self.DNI = self.input['DNI']
        self.DHI = self.input['DHI']
        self.theta_S_rad = deg2rad(self.input['theta_S_deg'])
        self.phi_S_rad = deg2rad(self.input['phi_S_deg'])
        self.albedo = self.input['albedo']
        self.ground_steps = self.input['ground_steps']
        self.module_steps = self.input['module_steps']
        self.angle_steps  = self.input['angle_steps']
        #Define variables derived from these base variables
        self.x_g_array = np.linspace(0,self.D,self.ground_steps)
        self.x_g_distance = self.D/(self.ground_steps-1) # distance between two points on x_g_array
        #self.l_array = np.linspace(0,self.L,self.module_steps) # OLD, changed on 26 April 2019!
        self.l_array = np.linspace(self.L/self.module_steps,self.L,self.module_steps)-0.5*self.L/self.module_steps
        # normal angle of the sun
        self.n_S = np.array([np.sin(self.theta_S_rad)*np.cos(-self.phi_S_rad),
                             np.sin(self.theta_S_rad)*np.sin(-self.phi_S_rad),
                             np.cos(self.theta_S_rad)])
        # initializing the results dictionary
        self.results = {}

    def module(self): # some functions and values for the PV module
        self.H_m = self.L*np.sin(self.theta_m_rad)
        self.e_m = np.array([np.cos(self.theta_m_rad),np.sin(self.theta_m_rad)]) # unit vector along the module
        self.n_m = np.array([-np.sin(self.theta_m_rad),np.cos(self.theta_m_rad)]) # normal to the module
        self.n_m_3D = np.array([self.n_m[0],0,self.n_m[1]]) # normal to the module

    # IRRADIANCE ON MODULE FROM THE SKY
    def calc_irradiance_module_sky_direct(self):
        temp_front = np.zeros(self.module_steps)
        temp_back  = np.zeros(self.module_steps)
        self.cos_alpha_mS = np.dot(self.n_S,self.n_m_3D) # cosine of angle between Sun and module normal
        angle_term = np.cos(self.theta_m_rad)-np.sin(self.theta_m_rad)*self.n_S[0]/self.n_S[2] # needed for calculating shadow on module
        if self.cos_alpha_mS > 0: #sunlight hits on front
            l_shadow = self.L-self.D/angle_term
            temp_front[l_shadow < self.l_array] = self.DNI*self.cos_alpha_mS # DMI = Direct Module Irradiance
        else: #sunlight hits on back
            l_shadow = self.L+self.D/angle_term
            temp_back[l_shadow < self.l_array] = self.DNI*np.abs(self.cos_alpha_mS)
        self.results['irradiance_module_front_sky_direct'] = temp_front
        self.results['irradiance_module_back_sky_direct']  = temp_back
        self.results['irradiance_module_front_sky_direct_mean'] = np.mean(temp_front)
        self.results['irradiance_module_back_sky_direct_mean']  = np.mean(temp_back)

    def calc_irradiance_module_sky_diffuse(self):
        alpha_2 = -np.pi/2.0
        for fb in ['front','back']:
            field_name = 'irradiance_module_' + fb + '_sky_diffuse'
            temp = np.zeros(self.module_steps)
            for i,l in enumerate(self.l_array):
                if fb == 'front':
                    vector_1 = (self.L-l)*self.e_m - np.array([self.D,0])
                if fb == 'back':
                    vector_1 = (self.L-l)*self.e_m + np.array([self.D,0])
                alpha_1 = self.alpha(vector_1,fb)
                temp[i] = self.DHI * (np.sin(alpha_1)-np.sin(alpha_2))/2.0
            self.results[field_name] = temp
            self.results[field_name + '_mean'] = np.mean(temp)

    # RADIANCE OF THE GROUND
    # radiance of the ground originating from direct skylight

    def calc_radiance_ground_direct(self):
        # x-coordinates of shadow:
        s_A = -self.H/self.n_S[2]
        x_A = s_A * self.n_S[0]
        s_B = (-self.H-self.L*np.sin(self.theta_m_rad))/self.n_S[2]
        x_B = self.L*np.cos(self.theta_m_rad)+s_B * self.n_S[0]

        flipp_mask = x_B < x_A
        x_A, x_B = np.where(flipp_mask, x_B, x_A),\
                   np.where(flipp_mask, x_A, x_B)

        length_shadow = x_B - x_A
        #if length_shadow < self.D: # only in this case direct Sunlight will hit the ground
        x_A_red = np.remainder(x_A, self.D) # reduce such that x_A_red is in [0,D]
        # x_B_red = x_B - x_A + x_A_red
        # calculate no of elements that are in the shadow
        illum_array_temp = np.greater.outer(self.x_g_array, length_shadow)
        # roll the beginning of the shadowed area to x_A_red
        illum_array_temp = np.roll(illum_array_temp,
                                   np.round(x_A_red/3*self.ground_steps).astype(int),
                                   axis=0)
        illum_array_temp = illum_array_temp.T
        try:
            illum_array_temp = ((illum_array_temp*self.DNI)*np.cos(self.theta_S_rad)[:, None])
        except:
            illum_array_temp = ((illum_array_temp*self.DNI)*np.cos(self.theta_S_rad))
        illum_array_temp[length_shadow>=self.D] = 0

        irradiance_ground_direct_received = illum_array_temp
        self.results['radiance_ground_direct_emitted'] = irradiance_ground_direct_received / np.pi * self.albedo

    def calc_radiance_ground_direct2(self):
        illum_array_temp = np.zeros(self.ground_steps)
        # x-coordinates of shadow:
        s_A = -self.H/self.n_S[2]
        x_A = s_A * self.n_S[0]
        s_B = (-self.H-self.L*self.e_m[1])/self.n_S[2]
        x_B = self.L*self.e_m[0]+s_B * self.n_S[0]
        if x_B < x_A: # Sun from back
            x_A_new = x_B
            x_B_new = x_A
            x_A = x_A_new
            x_B = x_B_new
        length_shadow = x_B - x_A
        if length_shadow < self.D: # only in this case direct Sunlight will hit the ground
            x_A_red = fmod_pos(x_A,self.D) # reduce such that x_A_red is in [0,D]
            x_B_red = x_B - x_A + x_A_red
            if x_B_red < self.D:
                for i,x in enumerate(self.x_g_array):
                    if (x < x_A_red) or (x > x_B_red): # Instead of  (x > x_B_red) in v0.1 (x > x_A_red)
                        illum_array_temp[i] = 1.0
            else:
                x_B_red = x_B_red - self.D # this was after the enumerate -> error!
                for i,x in enumerate(self.x_g_array):
                    if (x > x_B_red) and (x < x_A_red):
                        #print(x)
                        illum_array_temp[i] = 1.0
        irradiance_ground_direct_received = illum_array_temp * self.DNI * np.cos(self.theta_S_rad)
        self.results['radiance_ground_direct_emitted']  = irradiance_ground_direct_received / np.pi * self.albedo # division by pi converts irradiance into radiance assuming Lambertian scattering


    # functions for radiance of the ground originating from diffuse skylight
    def a_i(self,i,x_g): # Vector between points (x_g,-H) and A_i (lower end of i'th module)
        return np.array([i*self.D,0])- np.array([x_g,-self.H])

    def b_i(self,i,x_g): # Vector between points (x_g,-H) and B_i (upper end of i'th module)
        return self.L*self.e_m + np.array([i*self.D,0])- np.array([x_g,-self.H])

    # radiance of the ground originating from diffuse skylight
    # this method leads to the same results as the one developped earlier (v0.1) but is much faster.
    def calc_radiance_ground_diffuse(self):
        illum_array_temp = np.zeros(self.ground_steps)
        sin_zeta = np.zeros(4)
        sin_eta  = np.zeros(4)
        for i,x_g in enumerate(self.x_g_array):
            for j in range(4):
                a = self.a_i(j-1,x_g)
                b = self.b_i(j-1,x_g)
                sin_eta[j]  = a[0]/np.linalg.norm(a)
                sin_zeta[j] = b[0]/np.linalg.norm(b)

            #middle section:
            temp = min(sin_eta[2],sin_zeta[2])-sin_zeta[1]
            #front section
            if sin_zeta[0] < sin_eta[1]:
                temp = temp + sin_eta[1] - sin_zeta[0]
            # back section
            if max(sin_eta[2],sin_zeta[2]) < sin_zeta[3]:
                temp = temp + sin_zeta[3] - max(sin_eta[2],sin_zeta[2])

            illum_array_temp[i] = temp/2
        irradiance_ground_diffuse_received = illum_array_temp * self.DHI
        self.results['radiance_ground_diffuse_emitted']  = irradiance_ground_diffuse_received / np.pi * self.albedo # division by pi converts irradiance into radiance assuming Lambertian scattering

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
                    vector_1 = np.array([-self.D,0])-l*self.e_m
                if fb == 'back':
                    vector_1 = np.array([ self.D,0])-l*self.e_m
                x_g_1 = l*self.e_m[0] - (l*self.e_m[1]+self.H)/vector_1[1]*vector_1[0]
                if fb == 'front':
                    lower_index = int(round(x_g_1/self.x_g_distance))
                    upper_index = int(round(x_g_2/self.x_g_distance))
                if fb == 'back':
                    lower_index = int(round(x_g_2/self.x_g_distance))
                    upper_index = int(round(x_g_1/self.x_g_distance))
                index_array = np.arange(lower_index,upper_index+1)
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
            self.results[field_name] = (temp*self.results['radiance_ground_direct_emitted']).sum(axis=1)
            self.results[field_name + '_mean'] = np.mean(self.results[field_name])

    #irradiance on the module from the ground originating from diffuse skylight
    def calc_irradiance_module_ground_diffuse(self):
        for fb in ['front','back']:
            field_name = 'irradiance_module_' + fb + '_ground_diffuse'
            matrix = self.results['module_' + fb + '_ground_matrix']
            temp = np.sum(matrix, axis = 1)
            self.results[field_name] = (temp*self.results['radiance_ground_diffuse_emitted']).sum(axis=1)
            self.results[field_name + '_mean'] = np.mean(self.results[field_name])


class ModuleIllumination2:
    def __init__(self, Dict):
        self.dict = Dict
        self.variables()
        self.module()
        self.calc_irradiance_module_sky_diffuse()
        self.calc_irradiance_module_sky_direct()
        self.calc_radiance_ground_direct()
        self.calc_radiance_ground_diffuse()
        self.calc_irradiance_module_ground_direct()
        self.calc_irradiance_module_ground_diffuse()

    def variables(self): # Import all the variables
        self.L = self.dict['L']
        self.theta_m_deg = self.dict['theta_m_deg']
        self.theta_m_rad = deg2rad(self.dict['theta_m_deg'])
        self.H = self.dict['H']
        self.D = self.dict['D']
        self.DNI = self.dict['DNI']
        self.DHI = self.dict['DHI']
        self.theta_S_rad = deg2rad(self.dict['theta_S_deg'])
        self.phi_S_rad = deg2rad(self.dict['phi_S_deg'])
        self.albedo = self.dict['albedo']
        self.x_array = self.dict['x_array']
        self.ground_steps = self.dict['ground_steps']
        self.module_steps = self.dict['module_steps']
        self.angle_step_deg = self.dict['angle_step_deg']
        #Define variables derived from these base variables
        self.x_g_array = np.linspace(0,self.D,self.ground_steps)
        self.l_array = np.linspace(0,self.L,self.module_steps)

    def module(self): # some functions and values for the PV module
        self.k_m = np.tan(self.theta_m_rad) # k-value of module line
        self.y_m = self.k_m*fmod_pos_array(self.x_array,self.D) # function of the module line
        self.H_m = self.L*np.sin(self.theta_m_rad)
        self.e_m = np.array([np.cos(self.theta_m_rad),np.sin(self.theta_m_rad)]) # unit vector along the module
        self.n_m = np.array([-np.sin(self.theta_m_rad),np.cos(self.theta_m_rad)]) # normal to the module
        self.n_m_3D = np.array([self.n_m[0],0,self.n_m[1]]) # normal to the module

    # IRRADIANCE ON MODULE FROM THE SKY
    def calc_irradiance_module_sky_direct(self):
        # normal angle of the sun
        self.n_S = np.array([np.sin(self.theta_S_rad)*np.cos(-self.phi_S_rad),
                             np.sin(self.theta_S_rad)*np.sin(-self.phi_S_rad),
                             np.cos(self.theta_S_rad)])

        self.cos_alpha_mS = np.dot(self.n_S.T, self.n_m_3D) # cosine of angle between Sun and module normal

        self.dict['irradiance_module_front_sky_direct'] = \
            np.where(self.cos_alpha_mS > 0,
                     self.DNI*self.cos_alpha_mS,
                     0)

        self.dict['irradiance_module_back_sky_direct'] = \
            np.where(self.cos_alpha_mS < 0,
                     self.DNI*np.abs(self.cos_alpha_mS),
                     0)

    def calc_irradiance_module_sky_diffuse(self):
        alpha_2 = -np.pi/2.0
        for fb in ['front','back']:
            field_name = 'irradiance_module_' + fb + '_sky_diffuse'
            self.dict[field_name] = np.zeros(self.module_steps)
            for i,l in enumerate(self.l_array):
                if fb == 'front':
                    vector_1 = (self.L-l)*self.e_m - np.array([self.D,0])
                if fb == 'back':
                    vector_1 = (self.L-l)*self.e_m + np.array([self.D,0])
                alpha_1 = self.alpha(vector_1,fb)
                self.dict[field_name][i] = self.DHI * (np.sin(alpha_1)-np.sin(alpha_2))/2.0
            self.dict[field_name + '_mean'] = np.mean(self.dict[field_name])

    # RADIANCE OF THE GROUND
    # radiance of the ground originating from direct skylight
    def calc_radiance_ground_direct_(self):
        illum_array_temp = np.zeros(self.ground_steps)
        # x-coordinates of shadow:
        s_A = -self.H/self.n_S[2]
        x_A = s_A * self.n_S[0]
        s_B = (-self.H-self.L*self.e_m[1])/self.n_S[2]
        x_B = self.L*self.e_m[0]+s_B * self.n_S[0]
        if x_B < x_A: # Sun from back
            x_A_new = x_B
            x_B_new = x_A
            x_A = x_A_new
            x_B = x_B_new
        length_shadow = x_B - x_A
        if length_shadow < self.D: # only in this case direct Sunlight will hit the ground
            x_A_red = fmod_pos(x_A,self.D) # reduce such that x_A_red is in [0,D]
            x_B_red = x_B - x_A + x_A_red
            for i,x in enumerate(self.x_g_array):
                if x_B_red < self.D:
                    if x < x_A_red:
                        illum_array_temp[i] = 1.0
                    elif x > x_A_red:
                        illum_array_temp[i] = 1.0
                else:
                    x_B_red = x_B_red-self.D
                    if (x > x_B_red) & (x < x_A_red):
                        illum_array_temp[i] = 1.0
        self.irradiance_ground_direct_received = illum_array_temp * self.DNI * np.cos(self.theta_S_rad)
        self.radiance_ground_direct_emitted  = self.irradiance_ground_direct_received / np.pi * self.albedo # division by pi converts irradiance into radiance assuming Lambertian scattering

    def calc_radiance_ground_direct(self):
        # x-coordinates of shadow:
        s_A = -self.H/self.n_S[2]
        x_A = s_A * self.n_S[0]
        s_B = (-self.H-self.L*np.sin(self.theta_m_rad))/self.n_S[2]
        x_B = self.L*np.cos(self.theta_m_rad)+s_B * self.n_S[0]

        flipp_mask = x_B < x_A
        x_A, x_B = np.where(flipp_mask, x_B, x_A),\
                   np.where(flipp_mask, x_A, x_B)

        length_shadow = x_B - x_A
        #if length_shadow < self.D: # only in this case direct Sunlight will hit the ground
        x_A_red = np.remainder(x_A, self.D) # reduce such that x_A_red is in [0,D]
        # x_B_red = x_B - x_A + x_A_red
        # calculate no of elements that are in the shadow
        illum_array_temp = np.greater.outer(self.x_g_array, length_shadow)
        # roll the beginning of the shadowed area to x_A_red
        illum_array_temp = np.roll(illum_array_temp,
                                   np.round(x_A_red/3*self.ground_steps).astype(int),
                                   axis=0)
        illum_array_temp = illum_array_temp.T
        try:
            illum_array_temp = ((illum_array_temp*self.DNI)*np.cos(self.theta_S_rad)[:, None])
        except:
            illum_array_temp = ((illum_array_temp*self.DNI)*np.cos(self.theta_S_rad))
        illum_array_temp[length_shadow>=self.D] = 0

        self.irradiance_ground_direct_received = illum_array_temp
        self.radiance_ground_direct_emitted = self.irradiance_ground_direct_received / np.pi * self.albedo




if __name__ == '__main__':
    # Define the dictionary for the class
    InputDict = {
        'L': 1.650,# module length, standard is 1650 mm or 1960 mm
        'theta_m_deg': 52., # Angle of the module with respect to the ground (first guess optimal theta_m = latitude on Earth)
        'D': 3.000, # distance between modules
        'H': 0.500, # height of module base above ground
        'DNI': 1, # direct normal irradiance
        'DHI': 1, # diffuse horizontal irradiance
        'theta_S_deg': 30, # zenith of the Sun
        'phi_S_deg': 150, # azimuth of the Sun
        'albedo': 0.3, # albedo of the ground
        'ground_steps': 101, #number of steps into which irradiance on the ground is evaluated in the interval [0,D]
        'module_steps': 12, # SET THIS NUMBER HIGHER FOR REAL EVALUATION! (SUGGESTION 20) number of lengths steps at which irradiance is evaluated on the module
        'angle_steps': 180 # Number at which angle discretization of ground light on module should be set
    }

    df = pd.read_hdf('tmy_spec_seattle.h5', 'table')

    df = df.set_index('dt')

    ghi_column_filter = df.columns.str.contains('GHI')
    dni_column_filter = df.columns.str.contains('DNI')

    df['ghi_total'] = df.loc[:,ghi_column_filter].sum(axis=1)*10
    df['dni_total'] = df.loc[:,dni_column_filter].sum(axis=1)*10

    df.zenith = 90 - df.zenith

    df['diffuse_total'] = df['ghi_total']-df['dni_total']*np.cos(df.zenith/180*np.pi)
    df = df.loc[df.zenith < 90]


    InputDict['theta_S_deg'] = df.zenith
    InputDict['phi_S_deg'] = df.azimuth

    GI = ModuleIllumination(InputDict) # GroundIllumination

    asdf

    # =============================================================================
    # figure = plt.figure(figsize=(6,4))
    # plt.plot(GI.l_array,GI.irradiance_module_front_ground_diffuse)
    # plt.plot(GI.l_array,GI.irradiance_module_back_ground_diffuse)
    # plt.plot(GI.l_array,GI.irradiance_module_front_ground_direct)
    # plt.plot(GI.l_array,GI.irradiance_module_back_ground_direct)
    # figure.legend(['front_ground_diff','back_ground_diff','front_ground_dir','back_ground_dir'])
    # plt.show()
    # =============================================================================

    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array,GI.results['irradiance_module_front_sky_direct'],label='front')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_sky_direct'],label='back')
    plt.legend()
    plt.title('Direct Illumination from the Sky')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    plt.show()

    #dict['theta_S_deg'] = 30 # zenith of the Sun
    #dict['phi_S_deg'] = 135 # azimuth of the Sun
    #GI = ModuleIllumination(dict) # GroundIllumination
    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.x_g_array,GI.results['radiance_ground_diffuse_emitted'],label='diffuse ground')
    plt.plot(GI.x_g_array,GI.results['radiance_ground_direct_emitted'],label='direct ground')
    plt.legend()
    #plt.title('Diffuse illumination from the sky')
    plt.xlabel('position on ground (m)')
    plt.ylabel('radiance from ground (m$^{-2}$) (DHI = 1)')
    plt.show()

    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array,GI.results['irradiance_module_front_ground_diffuse'],label='front, diffuse')
    plt.plot(GI.l_array,GI.results['irradiance_module_front_ground_direct'],label='front, direct')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_ground_diffuse'],label='back, diffuse')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_ground_direct'],label='back, direct')
    plt.legend()
    plt.title('Illumination from the ground')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array,GI.results['irradiance_module_front_sky_diffuse'],label='sky front')
    plt.plot(GI.l_array,GI.results['irradiance_module_back_sky_diffuse'],label='sky back')
    plt.legend()
    plt.title('Diffuse illumination from the sky')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    plt.show()


    # =============================================================================
    # GI.update_zenith_azimuth(45, 155)
    #
    # figure = plt.figure(figsize=(6,4))
    # plt.plot(GI.l_array, GI.dict['irradiance_module_front_ground_diffuse'],label='front, diffuse')
    # plt.plot(GI.l_array, GI.dict['irradiance_module_front_ground_direct'],label='front, direct')
    # plt.plot(GI.l_array, GI.dict['irradiance_module_back_ground_diffuse'],label='back, diffuse')
    # plt.plot(GI.l_array, GI.dict['irradiance_module_back_ground_direct'],label='back, direct')
    # plt.legend()
    # plt.title('Illumination from the ground')
    # plt.xlabel('position on module (m)')
    # plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    # figure = plt.figure(figsize=(6,4))
    # plt.plot(GI.l_array, GI.dict['irradiance_module_front_sky_diffuse'],label='sky front')
    # plt.plot(GI.l_array, GI.dict['irradiance_module_back_sky_diffuse'],label='sky back')
    # plt.legend()
    # plt.title('Diffuse illumination from the sky')
    # plt.xlabel('position on module (m)')
    # plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    # =============================================================================

    # =============================================================================
    # figure = plt.figure(figsize=(6,4))
    # plt.plot(GI.l_array,GI.irradiance_module_front_sky_diffuse)
    # plt.plot(GI.l_array,GI.irradiance_module_back_sky_diffuse)
    # plt.show()
    # =============================================================================


# =============================================================================
#     from numpy import cross, eye, dot
#     from scipy.linalg import expm, norm
#
#     def M(axis, theta):
#         return expm(cross(eye(3), axis/norm(axis)*theta))
#
#     v1 = np.array([1,0,0])
#     v2 = np.array([0.8, 0.4, 0])
#     v2 = v2/np.linalg.norm(v2)
#     v_axis_1 = np.array([0,-1,0])
#     v_axis_2 = np.array([v2[1], -v2[0], 0])
#
#     axis_1 = np.cross(v1, v_axis_1)/np.linalg.norm(np.cross(v1, v_axis_1))
#     axis_2 = np.cross(v2, v_axis_2)/np.linalg.norm(np.cross(v2, v_axis_2))
#     theta_array = np.linspace(0, np.pi/2, 91)
#     distance = np.zeros_like(theta_array)
#
#     for i, theta in enumerate(theta_array):
#         v1_rot = np.dot(M(v_axis_1, theta), v1)
#         v2_rot = np.dot(M(v_axis_2, theta), v2)
#         distance[i] = np.arccos(np.dot(v1_rot, v2_rot))
#     plt.plot(theta_array, distance)
#
#     v, axis, theta = [3,5,0], [4,4,1], 1.2
#     M0 = M(axis, theta)
#     print(dot(M0,v))
# =============================================================================


