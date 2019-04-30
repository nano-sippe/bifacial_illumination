# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate
from scipy.integrate import quad
import time

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

    # functions for radiance of the ground originating from diffuse skylight
    def light_ray(self,psi,x_g): # inverse light ray originating at (x_g,-H) under direction psi
        k_g = np.tan(psi)
        d_g = -self.H - x_g*k_g
        return k_g*x_array+d_g

    def b_i(self,i,x_g): # Vector between points (x_g,-H) and B_i (upper end of i'th module)
        return self.L*self.e_m + np.array([i*self.D,0])- np.array([x_g,-self.H])

    def gamma_deg(self,b_i): # angle gamma corresponding to b_i
        cosine = b_i[0]/np.linalg.norm(b_i)
        return rad2deg(np.arccos(cosine))

    # radiance of the ground originating from diffuse skylight
    def calc_radiance_ground_diffuse(self):
        illum_array_temp = np.zeros(self.ground_steps)
        delta = self.angle_step_deg
        for i,x_g in enumerate(self.x_g_array):
            b__1 = self.b_i(-1,x_g)
            b_2  = self.b_i( 2,x_g)
            gamma_1 = self.gamma_deg(b__1)
            gamma_2 = self.gamma_deg(b_2)
            gamma_lo = np.floor(gamma_2)+.5
            gamma_hi = np.ceil(gamma_1)-.5
            gamma_dist = gamma_hi - gamma_lo + 1
            gamma_range = np.linspace(gamma_lo,gamma_hi,gamma_dist/delta)
            temp = 0
            for j,gamma in enumerate(deg2rad(gamma_range)):
                k_g = np.tan(gamma)
                d_g = -self.H - x_g*k_g
                y_g = k_g*self.x_array+d_g
                idx = []
                idx = np.where(np.logical_and(np.append(np.diff(np.sign(self.y_m - y_g)),0),(self.y_m < self.H_m)))
                if len(idx[0]) == 0:
                    temp = temp + np.sin(gamma)*np.sin(deg2rad(delta)/2.0)
            illum_array_temp[i] = temp
        self.irradiance_ground_diffuse_received = illum_array_temp * self.DHI
        self.radiance_ground_diffuse_emitted  = self.irradiance_ground_diffuse_received / np.pi * self.albedo # division by pi converts irradiance into radiance assuming Lambertian scattering

    # IRRADIANCE ON MODULE FROM THE SKY
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

    def radiance_ground_emitted_alpha(self,radiance,alpha,l,front_back):
        if front_back == 'front':
            x_alpha = l*self.e_m[0]+(l*self.e_m[1]+self.H)*np.tan(self.theta_m_rad+alpha)
        elif front_back == 'back':
            x_alpha = l*self.e_m[0]+(l*self.e_m[1]+self.H)*np.tan(self.theta_m_rad-alpha)
        else:
            print('ERROR! Value neither front nor back')
            x_alpha = 1./0.
        x_alpha = fmod_pos(x_alpha, self.D) # push the x-value to the [0,D] interval
        return pchip_interpolate(self.x_g_array,radiance,x_alpha)

    def calc_irradiance_module_ground_direct_from_matrix(self):
        self.calc_irradiance_module_ground_matrix()
        for fb in ['front', 'back']:
            field_name = 'irradiance_module_' + fb + '_ground_direct'
            irradiance = np.matmul(
                    self.dict['irradiance_{}_intensity_matrix'.format(fb)],
                    self.radiance_ground_direct_emitted.T).transpose(2,0,1)

            irradiance = irradiance*np.sin(deg2rad(np.arange(1,181)))*np.pi/180
            irradiance = irradiance.sum(axis=-1).T
            irradiance[0] = pchip_interpolate(self.l_array[1:],
                                              irradiance[1:],
                                              self.l_array[0],
                                              axis=0)
# =============================================================================
#             irradiance = irradiance.T
#             plt.plot(irradiance[2500])
# =============================================================================

            #calculate value at l = 0 via interpolation
            self.dict[field_name] = irradiance.T
            # calcualate mean values
            self.dict[field_name + '_mean'] = self.dict[field_name].mean(axis=-1)

    def calc_irradiance_module_ground_matrix(self):
        for fb in ['front','back']:
            intensity_matrix = np.zeros((len(self.l_array), 180, self.ground_steps))
            for i,l in enumerate(self.l_array): # remove 0th entry as otherwise the ray to calculate alpha1 would be parallel to x-axis!!!
                if i == 0:
                    continue
                module_pos_vec = l*self.e_m
                abs_pos_vec = module_pos_vec + np.array([0, self.H])

                if fb == 'front':
                    #lower and upper limit give the view range of the ground of a point on the module.
                    lower_limit = -(np.arccos(
                            module_pos_vec[1] / (module_pos_vec[0]+self.D))
                            *180/np.pi)\
                            .round().astype(int)
                    upper_limit = -int(90-self.dict['theta_m_deg'])
                    #effective angle is the incident angle of light for a certain viewing angle in the global reference system
                    effective_angle = abs(np.arange(lower_limit+1, upper_limit+1)-upper_limit)

                if fb == 'back':
                    upper_limit = (np.arccos(
                            module_pos_vec[1] / (self.D-module_pos_vec[0]))
                            *180/np.pi)\
                            .round().astype(int)
                    print(upper_limit)
                    lower_limit = -int(90-self.dict['theta_m_deg'])
                    effective_angle = np.arange(lower_limit+1, upper_limit+1)-lower_limit

                for j, angle in enumerate(range(lower_limit, upper_limit)):
                    angle_rad_low = deg2rad(angle)
                    angle_rad_high = deg2rad(angle+1)
                    #first calculate the minimum and maximum position (in meter) for a certain viewing angle
                    lower_distance = np.tan(angle_rad_low)*abs_pos_vec[1]+abs_pos_vec[0]
                    upper_distance = np.tan(angle_rad_high)*abs_pos_vec[1]+abs_pos_vec[0]
                    #then transform that to an index for the ground illumination array
                    lower_index = (lower_distance/self.D*self.ground_steps).round().astype(int)
                    upper_index = (upper_distance/self.D*self.ground_steps).round().astype(int)
                    indices = np.arange(lower_index, upper_index+1)%self.ground_steps
                    intensity_matrix[i,effective_angle[j], :] = \
                        np.histogram(indices, bins=np.arange(0, self.ground_steps+1), density=True)[0]

# =============================================================================
#             plt.plot((intensity_matrix[1]*self.radiance_ground_direct_emitted*np.pi/180).sum(axis=1))
#             plt.plot((intensity_matrix[1]*self.radiance_ground_diffuse_emitted*np.pi/180).sum(axis=1))
#             plt.plot(((intensity_matrix[1]*self.radiance_ground_direct_emitted*np.pi/180).sum(axis=1)*\
#                 np.sin(deg2rad(np.arange(1,181))))*np.pi/2)
# =============================================================================

            self.dict['irradiance_{}_intensity_matrix'.format(fb)] = \
                intensity_matrix*np.pi/2.0



    #irradiance on the module from the ground originating from direct skylight
    def calc_irradiance_module_ground_direct(self):
        radiance = self.radiance_ground_direct_emitted*np.pi/180
        for fb in ['front','back']:
            field_name = 'irradiance_module_' + fb + '_ground_direct'
            self.dict[field_name] = np.zeros(self.module_steps)
            for i,l in enumerate(self.l_array): # remove 0th entry as otherwise the ray to calculate alpha1 would be parallel to x-axis!!!
                if i == 0:
                    continue

                module_pos_vec = l*self.e_m
                abs_pos_vec = module_pos_vec + np.array([0, self.H])

                if fb == 'front':
                    #lower and upper limit give the view range of the ground of a point on the module.
                    lower_limit = -(np.arccos(
                            module_pos_vec[1] / (module_pos_vec[0]+self.D))
                            *180/np.pi)\
                            .round().astype(int)
                    upper_limit = -int(90-self.dict['theta_m_deg'])
                    #effective angle is the incident angle of light for a certain viewing angle in the global reference system
                    effective_angle = abs(np.arange(lower_limit+1, upper_limit+1)-upper_limit)

                if fb == 'back':
                    upper_limit = (np.arccos(
                            module_pos_vec[1] / (self.D-module_pos_vec[0]))
                            *180/np.pi)\
                            .round().astype(int)
                    print(upper_limit)
                    lower_limit = -int(90-self.dict['theta_m_deg'])
                    effective_angle = np.arange(lower_limit+1, upper_limit+1)-lower_limit

                radiance_tmp = np.zeros(upper_limit-lower_limit)

                for j, angle in enumerate(range(lower_limit, upper_limit)):
                    angle_rad_low = deg2rad(angle)
                    angle_rad_high = deg2rad(angle+1)
                    #first calculate the minimum and maximum position (in meter) for a certain viewing angle
                    lower_distance = np.tan(angle_rad_low)*abs_pos_vec[1]+abs_pos_vec[0]
                    upper_distance = np.tan(angle_rad_high)*abs_pos_vec[1]+abs_pos_vec[0]
                    #then transform that to an index for the ground illumination array
                    lower_index = (lower_distance/self.D*self.ground_steps).round().astype(int)
                    upper_index = (upper_distance/self.D*self.ground_steps).round().astype(int)
                    radiance_tmp[j] = np.take(radiance, np.arange(lower_index, upper_index+1), mode='warp').mean()

# =============================================================================
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111)
#                 ax.plot(effective_angle, radiance_tmp*np.sin(
#                         deg2rad(effective_angle)
#                         ))
#                 ax.set(ylabel='irradiance', xlabel='incident angle')
#                 plt.show()
# =============================================================================

                irradiance = (radiance_tmp*np.sin(
                        deg2rad(effective_angle)
                        )).sum()*np.pi/2.0

                self.dict[field_name][i] = irradiance
            #calculate value at l = 0 via interpolation
            self.dict[field_name][0] = pchip_interpolate(self.l_array[1:],self.dict[field_name][1:],self.l_array[0])
            # calcualate mean values
            self.dict[field_name + '_mean'] = np.mean(self.dict[field_name])

    #irradiance on the module from the ground originating from diffuse skylight
    # Der Code fuer diese Funktion und die Funktion oberhalb ist quasi derselbe. Ich habe sie getrennt, damit nicht immer alles berechent werden muss.
    def calc_irradiance_module_ground_diffuse(self):
        radiance = self.radiance_ground_diffuse_emitted*np.pi/180
        for fb in ['front','back']:
            field_name = 'irradiance_module_' + fb + '_ground_diffuse'
            self.dict[field_name] = np.zeros(self.module_steps)
            for i,l in enumerate(self.l_array): # remove 0th entry as otherwise the ray to calculate alpha1 would be parallel to x-axis!!!
                if i == 0:
                    continue

                module_pos_vec = l*self.e_m
                abs_pos_vec = module_pos_vec + np.array([0, self.H])

                if fb == 'front':
                    #lower and upper limit give the view range of the ground of a point on the module.
                    lower_limit = -(np.arccos(
                            module_pos_vec[1] / (module_pos_vec[0]+self.D))
                            *180/np.pi)\
                            .round().astype(int)
                    upper_limit = -int(90-self.dict['theta_m_deg'])
                    #effective angle is the incident angle of light for a certain viewing angle in the global reference system
                    effective_angle = abs(np.arange(lower_limit+1, upper_limit+1)-upper_limit)

                if fb == 'back':
                    upper_limit = (np.arccos(
                            module_pos_vec[1] / (self.D-module_pos_vec[0]))
                            *180/np.pi)\
                            .round().astype(int)
                    print(upper_limit)
                    lower_limit = -int(90-self.dict['theta_m_deg'])
                    effective_angle = np.arange(lower_limit+1, upper_limit+1)-lower_limit

                radiance_tmp = np.zeros(upper_limit-lower_limit)

                for j, angle in enumerate(range(lower_limit, upper_limit)):
                    angle_rad_low = deg2rad(angle)
                    angle_rad_high = deg2rad(angle+1)
                    #first calculate the minimum and maximum position (in meter) for a certain viewing angle
                    lower_distance = np.tan(angle_rad_low)*abs_pos_vec[1]+abs_pos_vec[0]
                    upper_distance = np.tan(angle_rad_high)*abs_pos_vec[1]+abs_pos_vec[0]
                    #then transform that to an index for the ground illumination array
                    lower_index = (lower_distance/self.D*self.ground_steps).round().astype(int)
                    upper_index = (upper_distance/self.D*self.ground_steps).round().astype(int)
                    radiance_tmp[j] = np.take(radiance, np.arange(lower_index, upper_index+1), mode='warp').mean()

# =============================================================================
#                 fig = plt.figure()
#                 ax = fig.add_subplot(111)
#                 ax.plot(effective_angle, radiance_tmp*np.sin(
#                         deg2rad(effective_angle)
#                         ))
#                 ax.set(ylabel='irradiance', xlabel='incident angle')
#                 plt.show()
# =============================================================================

                irradiance = (radiance_tmp*np.sin(
                        deg2rad(effective_angle)
                        )).sum()*np.pi/2.0

                self.dict[field_name][i] = irradiance
            #calculate value at l = 0 via interpolation
            self.dict[field_name][0] = pchip_interpolate(self.l_array[1:],self.dict[field_name][1:],self.l_array[0])
            # calcualate mean values
            self.dict[field_name + '_mean'] = np.mean(self.dict[field_name])

    def update_zenith_azimuth(self, zenith, azimuth):
        self.theta_S_rad = deg2rad(zenith)
        self.phi_S_rad = deg2rad(azimuth)
        self.calc_irradiance_module_sky_direct()
        self.calc_radiance_ground_direct()
        self.calc_irradiance_module_ground_direct()
        self.calc_irradiance_module_ground_diffuse()


if __name__ == '__main__':
    # Define the dictionary for the class
    dict = {
        'L': 1.650,# module length, standard is 1650 mm or 1960 mm
        'theta_m_deg': 52., # Angle of the module with respect to the ground (first guess optimal theta_m = latitude on Earth)
        'D': 3.000, # distance between modules
        'H': 0.500, # height of module base above ground
        'DNI': 1, # direct normal irradiance
        'DHI': 1, # diffuse horizontal irradiance
        'theta_S_deg': 89, # zenith of the Sun
        'phi_S_deg': 90, # azimuth of the Sun
        'albedo': 0.3, # albedo of the ground
        'x_array': np.linspace(-3,12,2001), # x-values for which the module and light ray functions are evaluated
        'ground_steps': 101, #number of steps into which irradiance on the ground is evaluated in the interval [0,D]
        'angle_step_deg': 0.1, # angular step at which diffuse illumination onto the ground is evaluated
        'module_steps': 6 # SET THIS NUMBER HIGHER FOR REAL EVALUATION! (SUGGESTION 20) number of lengths steps at which irradiance is evaluated on the module
    }

    GI = ModuleIllumination(dict) # GroundIllumination

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
    plt.plot(GI.l_array, GI.dict['irradiance_module_front_ground_diffuse'],label='front, diffuse')
    plt.plot(GI.l_array, GI.dict['irradiance_module_front_ground_direct'],label='front, direct')
    plt.plot(GI.l_array, GI.dict['irradiance_module_back_ground_diffuse'],label='back, diffuse')
    plt.plot(GI.l_array, GI.dict['irradiance_module_back_ground_direct'],label='back, direct')
    plt.legend()
    plt.title('Illumination from the ground')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')
    figure = plt.figure(figsize=(6,4))
    plt.plot(GI.l_array, GI.dict['irradiance_module_front_sky_diffuse'],label='sky front')
    plt.plot(GI.l_array, GI.dict['irradiance_module_back_sky_diffuse'],label='sky back')
    plt.legend()
    plt.title('Diffuse illumination from the sky')
    plt.xlabel('position on module (m)')
    plt.ylabel('Irradiance on module (m$^{-2}$) (DHI = 1)')

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


    from numpy import cross, eye, dot
    from scipy.linalg import expm, norm

    def M(axis, theta):
        return expm(cross(eye(3), axis/norm(axis)*theta))

    v1 = np.array([1,0,0])
    v2 = np.array([0.8, 0.4, 0])
    v2 = v2/np.linalg.norm(v2)
    v_axis_1 = np.array([0,-1,0])
    v_axis_2 = np.array([v2[1], -v2[0], 0])

    axis_1 = np.cross(v1, v_axis_1)/np.linalg.norm(np.cross(v1, v_axis_1))
    axis_2 = np.cross(v2, v_axis_2)/np.linalg.norm(np.cross(v2, v_axis_2))
    theta_array = np.linspace(0, np.pi/2, 91)
    distance = np.zeros_like(theta_array)

    for i, theta in enumerate(theta_array):
        v1_rot = np.dot(M(v_axis_1, theta), v1)
        v2_rot = np.dot(M(v_axis_2, theta), v2)
        distance[i] = np.arccos(np.dot(v1_rot, v2_rot))
    plt.plot(theta_array, distance)

    v, axis, theta = [3,5,0], [4,4,1], 1.2
    M0 = M(axis, theta)
    print(dot(M0,v))


