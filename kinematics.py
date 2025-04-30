from scipy.interpolate import CubicSpline
from scipy.integrate import quad, fixed_quad
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import materials
import json

with open('config.json') as file:
    config = json.load(file)
debug = config['debug']
def print_debug(message, variable):
    if debug == 'true':
        print(message)
        print(variable)


'''
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################



                        +--------------------------------------+
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
<---------------------  |                 |                    |              -------------------->
reverse alpha particles |     Backing     |   Radioisotope     |              f alpha particles
    going this way      |      Layer      |      Layer         |                 going this way
<---------------------  |                 |                    |              -------------------->
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        |                 |                    |
                        +--------------------------------------+

                              <---------------------
                              Direction of Travel
                              <---------------------

###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
'''







cache = {}


def energy_to_momentum(energy):
    return np.sqrt(2*energy*3727.3794118 + energy**2) #for alpha particles only. MeV input, MeV/c output


def compute_curves(material, alpha_decay_energy):
    csv_kinetic_energies = getattr(materials, material).kinetic_energy
    csv_ranges = getattr(materials, material).alpha_range

    
    csv_kinetic_energies = np.insert(csv_kinetic_energies, 0, 0)
    csv_ranges = np.insert(csv_ranges, 0, 0)

    #print_debug("csv_ranges:", csv_ranges)

    
    unadjusted = CubicSpline(csv_kinetic_energies, csv_ranges, extrapolate=False)

    #print_debug("alpha_energy:", alpha_decay_energy)
    x_offset = unadjusted(alpha_decay_energy)
    #print_debug("x_offset:", x_offset)
    csv_ranges = csv_ranges * (-1) + x_offset

    #print_debug("energies:", csv_kinetic_energies)
    #print_debug("ranges:", csv_ranges)


    momentums = energy_to_momentum(csv_kinetic_energies)


    #print_debug("Finite check for ranges:", np.isfinite(csv_ranges).all())
    #print_debug("Finite check for energies:" ,np.isfinite(csv_kinetic_energies).all())
    

    #Create splines
    experimental_energy_curve = CubicSpline(csv_ranges[::-1], csv_kinetic_energies[::-1], extrapolate=False, bc_type='natural')
    experimental_inverse_energy_curve = CubicSpline(csv_kinetic_energies, csv_ranges, extrapolate=False, bc_type='natural')
    momentum_curve = CubicSpline(csv_ranges[::-1], momentums[::-1], extrapolate=False, bc_type='natural') 
    inverse_momentum_curve = CubicSpline(momentums, csv_ranges, extrapolate=False, bc_type='natural') 

    #Send them a dictionary to be used later
    cache[f"{material}_momentum_curve"] = momentum_curve
    cache[f"{material}_inverse_momentum_curve"] = inverse_momentum_curve
    cache[f"{material}_energy_curve"] = experimental_energy_curve
    cache[f"{material}_inverse_energy_curve"] = experimental_inverse_energy_curve



   
    


   

def thrust(RL_material, BL_material, RL_grammage, BL_grammage,specific_activity):
    U = cache[f"{RL_material}_momentum_curve"]
    V = cache[f"{RL_material}_inverse_momentum_curve"]
    V_prime = V.derivative()
    

    G = cache[f"{BL_material}_momentum_curve"]
    H = cache[f"{BL_material}_inverse_momentum_curve"]
    G_prime = G.derivative()
    H_prime = H.derivative()
    

    max_range = V(0) #max range in RL material. Used to compute total specific activity
    specific_volumetric_activity = max_range * specific_activity

    
    def PDF_m1(momentum_1): # momentum distribution of all alpha particles coming out of the RL
        return np.abs( V_prime(momentum_1) ) / max_range
    
    
    def F_thrust():
        
        def inner_integral(theta):
            # y = f(x)
            # x = g(y)
            # PDF Y(y) = PDF X ( g(y) ) * | g'(y) |
            RL_path = RL_grammage / np.cos(theta)

            min_momentum = np.nan_to_num( U(RL_path), nan=0) #minimum momentum coming out of the BL
            max_momentum = U(0) # maximum momentum coming out of RL

            f = lambda x: PDF_m1(x) * x

            result, _ = fixed_quad(f, min_momentum, max_momentum,)
            return np.nan_to_num(result, nan=0)
        
        
        # Outer Integral

        # Simpfilied surface integration 1/2pi * integral(2pi * sin * cos * inner_integral)
        g = lambda theta: 0.5 * np.sin(2*theta) * inner_integral(theta)

        outer_integral = quad(g, 0 , 1.57, limit=200)
        final_result = 0.5 * specific_volumetric_activity * 5.344286E-22 * outer_integral[0]
        error = 0.5 * specific_volumetric_activity * 5.344286E-22 * outer_integral[1]

        #print_debug(f"F thrust: {final_result}")

        return final_result, error


        

        


        

    def R_thrust():
        
        def PDF_m2(momentum_2, theta): # momentum distribution of all alpha particles coming out of the BL
            # y = f(x)
            # x = g(y)
            # PDF Y(y) = PDF X ( g(y) ) * | g'(y) |
            BL_path = BL_grammage / np.cos(theta)
            

            
            probability = PDF_m1( G( H(momentum_2) - BL_path ) ) * np.abs( G_prime( H(momentum_2) - BL_path ) * H_prime(momentum_2) )

            return np.nan_to_num(probability, nan=0)
        



        #debug_counter = [0]

        def inner_integral(theta):
            #debug_counter[0] += 1
            #print_debug(f"Number: {debug_counter} angle(rad): {theta}")

            BL_path = BL_grammage / np.cos(theta)
            RL_path = RL_grammage / np.cos(theta)

            #print_debug(f"BL path: {BL_path}")
            #print_debug(f"RL path: {RL_path}")


            min_momentum = np.nan_to_num( G( BL_path + H( U(RL_path) ) ), nan=0 )
            max_momentum = np.nan_to_num( G(BL_path), nan=0) #maximum momentum coming out of the BL material

            #print_debug("min momentum:", min_momentum)
            #print_debug("max momentum:", max_momentum)

            f = lambda x, theta: PDF_m2(x, theta) * x

            

            result, _ = fixed_quad(f, min_momentum, max_momentum, args=(theta,))

            #print_debug(f"Number {debug_counter}, output: {result}")
            
            return np.nan_to_num(result, nan=0)
        
       
            

        
        

        if H(0) < BL_grammage:
            #print_debug("R Thrust: 0")
            return 0, 0
        else:
            max_angle = np.arccos( BL_grammage / H(0))

        #print_debug("H(0)", H(0))
        #print_debug("BL_grammage:", BL_grammage)
        #print_debug("max angle:", max_angle)

        
        
        # ---------Outer Integral-----------

        # Simpfilied surface integration 1/2pi * integral(2pi * sin * cos * inner_integral)
        g = lambda theta: 0.5 * np.sin(2*theta) * inner_integral(theta)

        outer_integral, error = quad(g, 0 , max_angle)
        final_result = 0.5 * specific_volumetric_activity * 5.344286E-22 * outer_integral
        error = 0.5 * specific_volumetric_activity * 5.344286E-22
        
        #print_debug(f"R Thrust: {final_result}")

        return final_result, error
    


    F_result, F_error = F_thrust()
    R_result, R_error = R_thrust()

    total_error = np.sqrt(F_error**2 + R_error**2)
    total_thrust = F_result - R_result
    return total_thrust, total_error, F_result, R_result




        

def acceleration_initial(RL_material, BL_material, RL_grammage, BL_grammage, decay_energy, specific_activity):
    
    compute_curves(RL_material, decay_energy)
    compute_curves(BL_material, decay_energy)
    force = thrust(RL_material, BL_material, RL_grammage, BL_grammage, specific_activity)[0]

    acceleration = 1000 * force/(RL_grammage + BL_grammage) #N/kg or m/s^2. Multiply by 1000 to convert from grams to kg
    #print_debug(f"RL: {RL_grammage}, BL: {BL_grammage}, thrust: {force} N, acceleration: {acceleration} m/s^2")
    return acceleration


'''
#### IN PROGRESS
def travel_distance_optimzation(RL_material, BL_material, RL_grammage, BL_grammage, decay_energy, specific_activity, travel_time, initial_velocity):
    compute_curves(RL_material, decay_energy)
    compute_curves(BL_material, decay_energy)
    k = 87.7 #decay constnat
    force = lambda t: thrust(RL_material, BL_material, RL_grammage, BL_grammage, specific_activity)[0] * np.exp(-k*t)#N/cm^2
    weight = BL_grammage + RL_grammage * 

'''













    







