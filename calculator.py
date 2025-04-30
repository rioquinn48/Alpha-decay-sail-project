import numpy as np

type = input("What do you want to find?   ")

if type == "momentum":
    input = input("Energy in Mev:   ")
    energy = float(input)
    momentum = np.sqrt(2*energy*3727.3794118+energy**2)
    print(f"Momentum: {momentum} MeV/c")
if type == "energy":
    input = input("Momentum in MeV/c:   ")
    momentum = float(input)
    energy = np.sqrt(momentum**2+3727.3794118**2)-3727.3794118
    print(f"Energy: {energy} MeV")
if type == "specific activity":
    half_life = float(input("Half Life in seconds:   "))
    mass = float(input("radisotope mass in Da:    "))
    specific_activity = (6.02214076E+23 * 0.6931471805) / ( half_life * mass)
    print(f"specific activity in Bq/g:  {specific_activity}")