import pandas as pd

class Material:
    def __init__(self, name, kinetic_energy, dedx, alpha_range, density, decay_energy, specific_activity, atomic_mass):
        self.name = name
        self.kinetic_energy = kinetic_energy
        self.dedx = dedx
        self.alpha_range = alpha_range
        self.density = density
        self.decay_energy = decay_energy
        self.specific_activity = specific_activity
        self.atomic_mass = atomic_mass
        

def load_mat_from_csv(mat):
    df = pd.read_csv(f"materials/{mat}.csv")

    material = Material(
        mat, 
        df['kinetic_energy'].tolist(),
        df['stopping_power'].tolist(), 
        df['alpha_range'].tolist(),
        df['density'][0],
        df['decay_energy'][0],
        df['specific_activity'][0],
        df['atomic_mass'][0]
        )
   
    return material



#enter the materials here
polonium = load_mat_from_csv("polonium") # data from Uranium need to be recalculated
kapton = load_mat_from_csv("kapton")
aluminum = load_mat_from_csv("aluminum")
plutonium_238 = load_mat_from_csv("plutonium_238")



