from scipy.optimize import minimize
import kinematics as kn
import json
import materials



'''
#################################################
#################################################
----------Optimization problem solving----------
'''


with open('config.json') as file:
    config = json.load(file)

RL_material = config['radioisotope_layer_material']
BL_material = config['backing_layer_material']
radionuclide_mass = getattr(materials, RL_material).atomic_mass
specific_activity = getattr(materials, RL_material).specific_activity
decay_energy = getattr(materials, RL_material).decay_energy
alpha_energy = decay_energy / ( 1 + 4.001506179127/radionuclide_mass )






def main():

    def objective(params):
        x, y = params    #x is RL, y is BL
        return -kn.acceleration_initial(RL_material, BL_material, x, y, alpha_energy, specific_activity)
    
    
    initial_guess = [0.002, 0.002]

    bounds = [(0.001,0.01), (0.001,0.01)]

    method = 'L-BFGS-B'

    options = {'disp': True, 'maxiter': 1000}

    result = minimize(objective, initial_guess, bounds=bounds, method=method, options=options)

    # Step 7: Print the results
    
    print(f"Optimized Parameters  RL: {10000*result.x[0]} g/m^2, BL: {10000*result.x[1]} g/m^2")
    print(f"Acceleration: {-result.fun} m/s^2")
    thrust, _, F_thrust, R_thrust = kn.thrust(RL_material, BL_material, result.x[0], result.x[1], specific_activity)
    print(f"Thrust: {thrust*1E+4} N/m^2")
    print(f"Forward Thrust {F_thrust*1E+4} N/m^2")
    print(f"Reverse Thrust {R_thrust*1E+4} N/m^2")


if __name__ == "__main__":
    main()

    

