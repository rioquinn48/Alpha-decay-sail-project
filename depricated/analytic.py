from scipy.interpolate import CubicSpline
from scipy.integrate import quad_vec
import numpy as np
import matplotlib.pyplot as plt
import materials



cache = {}
threshold = 0.001

def energy_to_momentum(energy):
    return np.sqrt(2*energy*3727.3794118+energy**2) #for alpha particles only. MeV input, MeV/c output

def compute_curves(material, alpha_decay_energy):
    csv_kinetic_energies = getattr(materials, material).kinetic_energy
    csv_ranges = getattr(materials, material).alpha_range

    
    csv_kinetic_energies = np.insert(csv_kinetic_energies, 0, 0)
    csv_ranges = np.insert(csv_ranges, 0, 0)

    unadjusted = CubicSpline(csv_kinetic_energies, csv_ranges, extrapolate=False)


    x_offset = unadjusted(alpha_decay_energy)
    csv_ranges = csv_ranges * (-1) + x_offset


    momentums = energy_to_momentum(csv_kinetic_energies)
    

    #Create splines
    experimental_energy_curve = CubicSpline(csv_ranges[::-1], csv_kinetic_energies[::-1], extrapolate=False)
    experimental_inverse_energy_curve = CubicSpline(csv_kinetic_energies, csv_ranges, extrapolate=False)
    momentum_curve = CubicSpline(csv_ranges[::-1], momentums[::-1], extrapolate=False) 
    inverse_momentum_curve = CubicSpline(momentums, csv_ranges, extrapolate=False) 
    
    #Send them a dictionary to be used later
    cache[f"{material}_momentum_curve"] = momentum_curve
    cache[f"{material}_inverse_momentum_curve"] = inverse_momentum_curve
    cache[f"{material}_experimental_energy_curve"] = experimental_energy_curve
    cache[f"{material}_experimental_inverse_energy_curve"] = experimental_inverse_energy_curve




def analytic(RL_material, BL_material, initial_momentum, RL_grammage, BL_grammage):
    ### Get the curves
    B = cache[f"{BL_material}_momentum_curve"]
    B_inverse = cache[f"{BL_material}_inverse_momentum_curve"]
    R_inverse = cache[f"{RL_material}_inverse_momentum_curve"]
    R_inverse_prime = R_inverse.derivative()
    B_anti_deriv = B.antiderivative()
    B_2nd_anti_deriv = B_anti_deriv.antiderivative()

    

    u = RL_grammage / R_inverse(initial_momentum) #upper bounds of integration

    a = B_inverse( abs( R_inverse_prime(initial_momentum) ) / range_max )

    upper = B_2nd_anti_deriv( BL_grammage / u + a ) - u * B_anti_deriv(BL_grammage / u + a)
    lower = B_2nd_anti_deriv( BL_grammage + a ) - B_anti_deriv(BL_grammage + a)

    integral = upper - lower #just the symbolic integration of the angles


    return integral


def compute_meshgrid(RL_material, BL_material, RL_grammage, BL_grammage,step_number, upper_bound, lower_bound):

    Z = np.zeros((len(RL_grammage),len(BL_grammage)))

    
    for i, RL in enumerate(RL_grammage):
        for j, BL in enumerate(BL_grammage):

            

            Z[i,j] = analytic(RL_material, BL_material, 100, RL, BL)
    
    return Z


def momentum_pdf(material, cutoff_momentum, path_length, momentum_values):

    U = cache[f"{material}_momentum_curve"]
    V = cache[f"{material}_inverse_momentum_curve"]
    V_prime = V.derivative()

    
    mask = V(momentum_values) < path_length # Create a boolean mask

    max_range = V(0)
    print(max_range)

    # Initialize PDF to zeros
    pdf = np.zeros_like(momentum_values)

    # Apply the formula only where p < max_val
    pdf[mask] = (1 / max_range) * np.abs(V_prime(momentum_values[mask]))

    return pdf


def R_total_momentum_released(RL_material, BL_material, BL_grammage_values, RL_grammage_values, N): #For a given BL grammage, this creates the PDF of exit momentum based on a standard boundry momentum curve from RL
    np.set_printoptions(linewidth=1000, suppress=True, precision=10)
    
    U = cache[f"{RL_material}_momentum_curve"]
    V = cache[f"{RL_material}_inverse_momentum_curve"]
    V_prime = V.derivative()

    G = cache[f"{BL_material}_momentum_curve"]
    H = cache[f"{BL_material}_inverse_momentum_curve"]
    H_prime = H.derivative()
    G_prime = G.derivative()

    max_range = V(0) #max range in RL material
    max_momentum = U(0) #maximum momentum coming out of the RL material
    momentum_2_values = np.linspace(0,max_momentum, N) # Create N intervals of momentum between 0 and max momenentum. This will be rows


        


    def PDF_m1(momentum_1): # momentum distribution of all alpha particles coming out of the RL
        return np.abs( V_prime(momentum_1) ) / max_range
    
    def PDF_m2(momentum_2,BL_grammage_values): # momentum distribution of all alpha particles coming out of the BL
        # y = f(x)
        # x = g(y)
        # PDF Y(y) = PDF X ( g(y) ) * | g'(y) |
        return PDF_m1( G( H(momentum_2) - BL_grammage_values ) ) * np.abs( G_prime( H(momentum_2) - BL_grammage_values ) * H_prime(momentum_2) )
    



    
    #Create a grid for the outputs with BL_grammage columns and momentum_2 rows. Cell value is probability of that momentum
    grid = np.zeros((N,len(BL_grammage_values)))
    for index, BL_grammage in enumerate(BL_grammage_values):
        grid[:, index] = PDF_m2(momentum_2_values, BL_grammage) * momentum_2_values #For this column of BL we will have a proability for each row of momentum

    intervals = np.nan_to_num(grid, nan=0) # get rid of any nan values due to the splines domain
    # intervals. BL grammage as columns. Momentum as rows. Cells are momentum * probability of that momentum

    print("Momentum rows, BL grammage columns. Output is momentum * pdf(momentum) cells")
    print(intervals)

    
    




    min_momentum = np.zeros((len(RL_grammage_values),len(BL_grammage_values)))
    # Find the minimum momentum coming out of the BL for a given RL
    for i_min_mom_loop, RL_min_mom_loop in enumerate(RL_grammage_values):

        min_boundry_momentum = U(RL_min_mom_loop)  # Find the smallest momentum coming out of the RL; the bottom cutoff for the distribution

        for j_min_mom_loop, BL_min_mom_loop in enumerate(BL_grammage_values):

            min_momentum[ i_min_mom_loop, j_min_mom_loop ] = G( BL_min_mom_loop + H(min_boundry_momentum) ) #Find the final momentem as it exits the BL 

    min_momentum = np.nan_to_num(min_momentum, nan=0)
    #Lower bound for integration in terms of momentum

    print("RL grammage rows, BL grammage columns. Output is minimum momentum.")
    print(min_momentum)








    
    total_momentum_released = np.zeros((len(RL_grammage_values),len(BL_grammage_values)))
    #Calculate the total momentem released per decay for RL and BL values
    for BL_index_loop_3, BL_grammage_loop_3 in enumerate(BL_grammage_values):
            loop_3_f_p_p = intervals[:,BL_index_loop_3] #Takes all the momentum values for a single BL column from intervals and flip it upside down
            for RL_index_loop_3, RL_grammage_loop_3 in enumerate(RL_grammage_values):
                


            
            
            #Find the index to cut off momentum values at
        
            total_momentum_released[RL_index_loop_3, :] = np.trapezoid(intervals,  )
            
    print("RL grammage rows, BL grammage columns. Output is total momentum released.")
    print(total_momentum_released)
    return total_momentum_released
    
        
    





    


    





        


        




def R_thrust(RL_material, BL_material, RL_grammage, BL_grammage, specific_activity, decay_energy):
    U = cache[f"{RL_material}_momentum_curve"]
    V = cache[f"{RL_material}_inverse_momentum_curve"]
    V_prime = V.derivative()

    G = cache[f"{BL_material}_momentum_curve"]
    H = cache[f"{BL_material}_inverse_momentum_curve"]
    H_prime = H.derivative

    #create find upper and lower bounds for momentum
    max_momentum = np.nan_to_num(G(BL_grammage), nan=0) #this creates the upper bound off momentum for alpha particle momentum being emitted from the backing layer

    max_momentum_grid = np.tile(max_momentum, (len(RL_grammage),1))

    min_momentum = np.zeros((len(BL_grammage),len(RL_grammage)))

    for j, value_RL in enumerate(RL_grammage):
        min_boundry_momentum = U(value_RL)  # Find the smallest momentum coming out of the RL for a given grammage
        for i, value_BL in enumerate(BL_grammage):
            min_momentum[i, j] = G(value_BL + H(min_boundry_momentum)) #Find the final momentem as it exits the BL 

    min_momentum_grid = np.nan_to_num(min_momentum, nan=0)

    print(f" max momentum {max_momentum_grid}")
    print(f" min momentum {min_momentum_grid}")






    










    

def monte_carlo_double(function, A, x_upper, x_lower, y_upper, y_lower, N):

    a = np.array(A)
    output = np.zeros_like(a)

    for index, a_value in enumerate(a):
        
        x = np.random.uniform(x_lower, x_upper, N)
        y = np.random.uniform(y_lower, y_upper, N)

        func_values = function(x, y, a_value)
    
        mean_func_value = np.mean(func_values)

        area = np.float64((x_upper - x_lower) * (y_upper - y_lower))

        integral = area * mean_func_value
        output[index] = integral
    
    return output

def F_thrust(material, decay_momentum, specific_activity, RL_grammage, step_number):

    V = cache[f"{material}_inverse_momentum_curve"]
    max_range = V(decay_momentum)
    

    def f(x, y, a):
        return np.sin(y) * np.cos(y) * momentum_pdf(material, decay_momentum, (a / np.cos(y)), x)

    impulse_integrals = monte_carlo_double( f, RL_grammage, decay_momentum, 0, np.pi/2, 0, step_number )
    
    thrust = impulse_integrals * specific_activity * max_range * getattr(materials, material).density * 5.344286E-22 #MeV/c/s/cm^2 to N/cm^2
    print(thrust)

    return thrust
    

'''
###################################################################################################################################################################
###################################################################################################################################################################
'''

def graph1D(x_vals, y_vals,title,x_label,y_label,inverted_x=False, inverted_y=False):
    plt.plot(x_vals, y_vals)

    if inverted_x:
        plt.gca().invert_xaxis()
    if inverted_y:
        plt.gca().invert_yaxis()

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()



def graph2D(x_vals,y_vals,Z_meshgrid):
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = Z_meshgrid

    plt.contour(X,Y,Z, cmap="viridis")
    plt.colorbar
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Contour Plot")
    plt.show()



    
def run(RL_material, BL_material):
    compute_curves(RL_material)
    compute_curves(BL_material)
    compute_antiderivatives(RL_material)
    compute_antiderivatives(BL_material)

    
    RL_grammage = np.linspace(0,0.01,100)
    BL_grammage = np.linspace(0,0.01,100)
    integration_steps = 100

    initial_momentum = 200
    cutoff_momentum = 1

    Z = compute_meshgrid(RL_material, BL_material, RL_grammage, BL_grammage, integration_steps,initial_momentum,cutoff_momentum)
    
    
    print(Z)
    graph2D(BL_grammage, RL_grammage, Z)



def test1(BL_material,decay_energy):

    
    x = np.linspace( 0, 0.01, 1000 )
    compute_curves(BL_material, decay_energy)
    
    
    experimental = cache[f"{BL_material}_experimental_energy_curve"]

    y = experimental(x)

    print(experimental(0))
    graph1D(x,y,"","","")

    

    
    



def test2(material):

    
    
    compute_curves(material)

   
    RL_grammage = np.linspace(0,0.02,100)

    output = F_thrust(material, 198.84242948898003, 1.67E+14, RL_grammage, 100000) 

    graph1D(RL_grammage, output, f"Thrust vs RL grammage for {material}", "RL_grammage", "forward thrust")





def test3(material, decay_energy, specific_activity):
    compute_curves(material, decay_energy)
    path_grammage = 0.01

    initial_momentum = energy_to_momentum(decay_energy)
    
    p = np.linspace(0,initial_momentum,1000)

    probability = momentum_pdf(material, initial_momentum, path_grammage, p)

    area = np.trapezoid(probability, p)
    print(f"area: {area}")

    graph1D(
            p, 
            probability, 
            f"Momentum probability distribution for {path_grammage} g/cm^2 of {material}", 
            "Momentum (MeV/c)",
            "Probability (c/MeV)"
            )
    
def test4(BL_material, RL_material, decay_energy, N):
    
    compute_curves(BL_material, decay_energy)
    compute_curves(RL_material, decay_energy)

    BL_grammage_values = np.linspace(0.001,0.01, 10)
    RL_grammage_values = np.linspace(0.001,0.01, 10)


    Z = R_total_momentum_released(RL_material, BL_material, BL_grammage_values, RL_grammage_values, N)
   

    

    
    
    
    



test4("kapton", "polonium", 5.3, 100)