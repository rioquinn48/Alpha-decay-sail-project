from scipy.interpolate import CubicSpline
from scipy.integrate import odeint, quad, cumulative_simpson, cumulative_trapezoid
from materials import load_mat_from_csv
import numpy as np
import multiprocessing

import matplotlib.pyplot as plt
import time
import cProfile
import pstats



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






Definitions:
Energy (MeV) - kinetic energy of the alpha particle MeV
Momentum (MeV/c) - momentum of the alpha particle
Impulse (MeV/c) - momentum in the forward direction
mDistance (g/cm^2) - mass-normalized distance; distance (cm) multiplied by material density(g/cm^3)
Grammage (g/cm^2) - layer thickness (cm) multiplied by material density(g/cm^3)

Abbreviations
m... (mDistance, mDepth, mLocation, etc.) - mass-normalized value
RL - radioisotope layer
BL - backing layer
F - forward(for the alpha particles); emitted in the correct direction, opposite to the direction of travel
R - reverse(for the alpha particles); emitted in the wrong direction, slowing the sail down
_________ - shorthand for "as a function of"

  

###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################
'''




'''
Here are the most basic functions. They create the dE/dx, E(X), 
their inverses and the momentum curves for each material.
'''

threshold = 0.001 #in MeV. Set the minimum energy thershold below which we ignore

cache = {} #Create a cache to store spline functions so they don't have to be recalculated


def compute_curves(material):
    data = load_mat_from_csv(material) #load the material data from its .csv in the materials file

    stopping_power = CubicSpline(data.kinetic_energy, data.dedx) #create a dE/dx curve for this material

    def dedx(mDistance, e):
        return (-1)*stopping_power(e)

    e0 = 6
    mDistance = np.linspace(0, 0.03, 3000) #Make sure that this is greater than the max range of all materials in g/cm^2
    energy = odeint(dedx, y0 = e0, t=mDistance, tfirst=True)[:,0] #Integrate dE/dx using the odeint function to find E(x)
    

    idx = np.argmax(energy < threshold) #Find the first occurance where the energy value < threshold in E(x)
    if idx != 0:
        mDistance = mDistance[:idx]
        energy = energy[:idx] #Chop both lists there
    
    

    momentum = np.sqrt(2*energy*3727.3794118+energy**2) #convert from energy to momentum

    #Create splines
    energy_curve = CubicSpline(mDistance, energy, extrapolate=False) 
    inverse_energy_curve = CubicSpline(energy[::-1], mDistance[::-1], extrapolate=False) 
    momentum_curve = CubicSpline(mDistance, momentum, extrapolate=False) 
    inverse_momentum_curve = CubicSpline(momentum[::-1], mDistance[::-1], extrapolate=False) 
    inverse_momentum_derivative = inverse_momentum_curve.derivative()
    #Send them a dictionary to be used later
    cache[f"{material}_stopping_power"] = stopping_power
    cache[f"{material}_inverse_momentum_derivative"] = inverse_momentum_derivative
    cache[f"{material}_energy_curve"] = energy_curve
    cache[f"{material}_inverse_energy_curve"] = inverse_energy_curve
    cache[f"{material}_momentum_curve"] = momentum_curve
    cache[f"{material}_inverse_momentum_curve"] = inverse_momentum_curve
    





def momentum___________mDistance(material, mDistance, initial_momentum):
    if f"{material}_inverse_momentum_curve" in cache: #check if the curves are in the cache

        inverse_momentum_curve = cache[f"{material}_inverse_momentum_curve"]
        x_offset = inverse_momentum_curve(initial_momentum)

        momentum_curve = cache[f"{material}_momentum_curve"]
        final_momentum = np.nan_to_num(momentum_curve(mDistance+x_offset))
        return final_momentum
    
    else:
        compute_curves(material)
        return momentum___________mDistance(material, mDistance, initial_momentum)
    

def energy___________mDistance(material, mDistance, initial_energy):
    if f"{material}_inverse_energy_curve" in cache: #Check if the curves are in the cache

        inverse_energy_curve = cache[f"{material}_inverse_energy_curve"]
        x_offset = inverse_energy_curve(initial_energy)

        energy_curve = cache[f"{material}_energy_curve"]
        final_energy = np.nan_to_num(energy_curve(mDistance+x_offset))  #convert numbers outside of spline to 0
        return final_energy
    else:                   #if they aren't then create them and then put them rerun the function
        compute_curves(material)
        return energy___________mDistance(material, mDistance, initial_energy)
        
'''
########
########
Just some conversion functions
########
########
'''


def energy_to_momentum(rest_mass, energy):
    if rest_mass == "alpha":
        rest_mass = 3727.3794118
    momentum = np.sqrt(2*energy*rest_mass+energy**2)
    return momentum

def distance_um_to_mDistance(um, density):
    cm2g = 0.0001/density * um
    return cm2g

def mDistance_to_distance_um(cm2g, density):
    um = 10000*density*cm2g
    return um

def array_to_meshgrid(array):

    # Step 1: Extract unique x and y values
    x_unique = np.unique(array[:, 0])
    y_unique = np.unique(array[:, 1])

    # Step 2: Create meshgrid
    X, Y = np.meshgrid(x_unique, y_unique, indexing="ij")  # "ij" keeps (x, y) order

    # Step 3: Reshape Z to match the grid shape
    Z = array[:, 2].reshape(len(x_unique), len(y_unique))  # Shape matches X, Y
    return X, Y, Z


'''
###################################################################################################################################################################
###################################################################################################################################################################
Here are the more advanced functions. They calculate 
average momentum of alpha particles as they are emitted 
in all directions at different depths.
###################################################################################################################################################################
###################################################################################################################################################################
'''
#Computation medium: quad N times

# Add timing decorator to key functions
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper

'''
###################################################################################################################################################################
###################################################################################################################################################################
'''

@timing_decorator
def F_impulse___________decay_mLocation(material, mDistance, initial_momentum): #Gives the average forward impulse generated by one decay as a function of depth
    #mLocation is mass-normalized distance from the top of the radioisotope layer
    #can only handle lists!
    y_values = []
    
    for x_value in mDistance:

        def y(x):
            return 0.5*np.sin(2*x)*momentum___________mDistance(material, (x_value/(np.cos(x))), initial_momentum)
        average_decay_impulse = quad(y, 0, 1.5)[0]
        y_values.append(average_decay_impulse)

    return y_values

'''
###################################################################################################################################################################
###################################################################################################################################################################
'''
#Computation Medium: cumulative mean N times
@timing_decorator
def average_cumulative_F_impulse___________radioisotype_layer_grammage(material, grammage_values, initial_momentum): #Grammage in g/cm^2
    
    
    f = F_impulse___________decay_mLocation(material, grammage_values, initial_momentum)
    F = cumulative_simpson(f, x=grammage_values, initial=initial_momentum)
    
    # Vectorized calculation of cumulative mean
    indices = np.arange(1, len(F) + 1)
    cumulative_mean_list = F / indices
    
    return cumulative_mean_list

'''
###################################################################################################################################################################
###################################################################################################################################################################
'''


def F_thrust___________radioisotope_layer_grammage(material, RL_grammage, decay_energy, specific_activity): #in N/cm^2
    
    initial_momentum = energy_to_momentum("alpha",decay_energy)#convert to momentum

    impulses = average_cumulative_F_impulse___________radioisotype_layer_grammage(material, RL_grammage, initial_momentum)
    
     #Thrust/cm^2 = Impulse * g/cm^2 * specific_activity x
    thrust = impulses * RL_grammage *  specific_activity * 5.344286E-22 #1MeV/c = 5.344286E-22kgm/s
        

    return thrust


'''
###################################################################################################################################################################
###################################################################################################################################################################
'''


def R_momentum___________decay_mLocation_AND_angle(material, decay_mLocation, polar_angle, initial_momentum): 
    #Gives final energy as a function of mass-normalized depth AND angle from normal
    
    x = decay_mLocation/(np.cos(polar_angle))

    energy = momentum___________mDistance(material, x, initial_momentum)
    return energy

'''
###################################################################################################################################################################
###################################################################################################################################################################
'''



def R_impulse___________BL_grammage_AND_decay_angle_AND_decay_mLocation(RL_material, BL_material, BL_grammage, decay_event_mLocation, initial_momentum, angle_from_normal):
    
    
    momentum_at_boundry = R_momentum___________decay_mLocation_AND_angle(RL_material, decay_event_mLocation, angle_from_normal, initial_momentum)
    
    momentum_at_exit = momentum___________mDistance(BL_material, BL_grammage, momentum_at_boundry)

    return momentum_at_exit*np.cos(angle_from_normal)

'''
###################################################################################################################################################################
###################################################################################################################################################################
'''

def R_impulse__________BL_grammage_AND_decay_mLocation(RL_material, BL_material, BL_grammage, decay_event_mLocation, initial_momentum):
    
    # Integrate for over all angles
    def f(theta, BL, RL):   
        return R_impulse___________BL_grammage_AND_decay_angle_AND_decay_mLocation(RL_material,BL_material, BL, RL, initial_momentum, theta) 

    BL_array = np.array(BL_grammage)
    RL_array = np.array(decay_event_mLocation)
    Z = np.zeros((len(RL_array),len(BL_array))) #Creates a 2D array
    # decay_event_mLocation is rows
    # BL_grammage is columns

     # Define a function to compute the result for each grid point
    def compute_integral(i, RL, BL_array, RL_array):
        row_result = []
        for j, BL in enumerate(BL_array):  # Iterate over the columns (BL_grammage)
           
            row_result.append(result)
        return i, row_result
    
    # Parallel processing using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        # For each RL value (iterate over rows), call compute_integral in parallel
        futures = [executor.submit(compute_integral, i, RL, BL_array, RL_array) for i, RL in enumerate(RL_array)]
        
        for future in futures:
            i, row_result = future.result()  # Get results from the future
            Z[i, :] = row_result  # Fill in the results for that row
    
    return Z 
'''
###################################################################################################################################################################
###################################################################################################################################################################
'''

def R_thrust__________BL_grammage_AND_RL_grammage(RL_material, BL_material, BL_grammage: np.ndarray, RL_grammage: np.ndarray, decay_energy, specific_activity):
    initial_momentum = energy_to_momentum("alpha",decay_energy)
    meshgrid = R_impulse__________BL_grammage_AND_decay_mLocation(RL_material, BL_material, BL_grammage, RL_grammage, initial_momentum)
    
    for i in range(len(meshgrid[1])): #iterating over each column of BL_grammage
        column = meshgrid[:,i]
        cum_sum_values = cumulative_simpson(column, x=RL_grammage, initial=column[0]) #Calculate the cumulative sums
        
        # Vectorized calculation of cumulative mean
        indices = np.arange(1, len(cum_sum_values) + 1)
        cumulative_mean_list = cum_sum_values / indices
        
    
        #Thrust/cm^2 = Impulse * g/cm^2 * specific_activity x
        thrust = cumulative_mean_list * RL_grammage *  specific_activity * 5.344286E-22 #1MeV/c = 5.344286E-22kgm/s
        
        meshgrid[:,i] = thrust
        

    return meshgrid

        
'''
###################################################################################################################################################################
###################################################################################################################################################################
'''

@timing_decorator
def total_thrust__________BL_grammage_AND_RL_grammage(RL_material, BL_material, BL_grammage, RL_grammage, decay_energies, specific_activities):

    Z = np.zeros((len(RL_grammage),len(BL_grammage)))
    for index, decay_energy in enumerate(decay_energies): #iterate for different energies
        R_meshgrid = (-1) * R_thrust__________BL_grammage_AND_RL_grammage(RL_material, BL_material, BL_grammage, RL_grammage, decay_energy, 0.5*specific_activities[index])
        F = np.array(F_thrust___________radioisotope_layer_grammage(RL_material, RL_grammage, decay_energy, 0.5*specific_activities[index]))
        
        ##########
        #Debug prints
        '''
        for column in range(len(R_meshgrid[1,:])):
            print(f"Column: {column}")
            print(f"For BL thickness: {BL_grammage[column]}")
            print(f"Forward thrust: {F}")
            print(f"Reverse thrust: {R_meshgrid[:,column]}")
            print("\n")
        
        ##########
        '''
        
        Total = R_meshgrid + F.reshape(-1,1) #transpose the F thrust, then add the F and R thrust
        Z = Z + Total #add to total thrust
        
    return Z



'''
###################################################################################################################################################################
###################################################################################################################################################################
'''

def monte_carlo_integration(f, a, b, num_samples=100000):
    
    theta_samples = np.random.uniform(a, b, num_samples) # Randomly sample points for theta

   
    f_values = np.array([f(theta) for theta in theta_samples]) # Evaluate the function at sampled points

    
    return (b - a) * np.mean(f_values) # Return the mean value times the width of the range (b - a) for an estimate of the integral


'''
###################################################################################################################################################################
###################################################################################################################################################################
'''





RL_grammage = np.linspace(0,0.01, 30)
BL_grammage = np.linspace(0,0.01, 30)
'''
my_study_backing= np.array([0.0048582,0.0051281,0.005398,0.0056679])
my_study_radioisotope = np.array([0.0004965,0.000993,0.0014895,0.001986,0.0024825,0.002979,0.0034755,0.003972,0.0044685,0.004965,0.0054615,0.005958,0.0064545,0.006951,0.0074475,0.007944,0.0084405,0.008937,0.0094335,0.00993,0.0104265,0.010923,0.0114195,0.011916,0.0124125,0.012909,0.0134055,0.013902,0.0143985,0.014895,0.0153915,0.015888,0.0163845,0.016881,0.0173775,0.017874,0.0183705,0.018867,0.0193635,0.01986])


china_study_backing = np.array([7e-4, 7.4101519E-4, 8E-4])
china_study_RL_thickness = np.linspace(5.6,22.4,30)
china_study_RL_grammage = china_study_RL_thickness*0.0001*9.25
#Z = total_thrust__________BL_grammage_AND_RL_grammage("plutonium_238", "aluminum", BL_grammage, RL_grammage, [5.4493, 5.4563], [3.60315E+11, 1.46327E+11])

#Z = total_thrust__________BL_grammage_AND_RL_grammage("plutonium_238","aluminum",my_study_backing, my_study_radioisotope,[5.4493, 5.4563, 5.3571],[3.60315E+11,1.46327E+11,5.27488E+08])
'''
# Profile the entire calculation
def run_calculation():
    return total_thrust__________BL_grammage_AND_RL_grammage(
        "polonium", "aluminum",
        BL_grammage, RL_grammage,
        [5.3], [1.67E+14]
    )

# Run with profiler
print("Running profiler...")
profiler = cProfile.Profile()
profiler.enable()
Z = run_calculation()
profiler.disable()

# Print detailed stats
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)  # Show top 20 time-consuming functions



'''
###################################################################################################################################################################
###################################################################################################################################################################
'''


def momentum_pdf(material, momentum, path_mLength, max_range):
    v = cache[f"{material}_inverse_momentum_curve"]
    v_prime = cache[f"{material}_inverse_momentum_derivative"]
    
    if v(momentum) < path_mLength:
        return abs(v_prime(momentum))/max_range
    else:
        return 0

    
'''
###################################################################################################################################################################
###################################################################################################################################################################
'''

def impulse_pdf_monte_carlo(material, impulse, angle, grammage, max_range):
    v = cache[f"{material}_inverse_momentum_curve"]
    v_prime = cache[f"{material}_inverse_momentum_derivative"]
    
    if v(impulse/np.cos(angle)) < grammage/np.cos(angle):
        return abs(v_prime(impulse/np.cos(angle)) * np.cos(angle))/max_range
    else:
        return 0
    
'''
###################################################################################################################################################################
###################################################################################################################################################################
'''  

    # Parallel Monte Carlo using multiprocessing
def parallel_monte_carlo(func, x, y, num_samples, num_processes):
    # Split the work into chunks
    chunk_size = num_samples // num_processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(monte_carlo_integration, [(func, x, y, chunk_size) for _ in range(num_processes)])
    # Average the results from each process
    return np.mean(results)


'''
###################################################################################################################################################################
###################################################################################################################################################################
'''  



 




        
        








        










