import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import pandas as pd

#Get x and y values from csv
df = pd.read_csv('materials/plutonium_238.csv')
x_list = df['kinetic_energy'].tolist()
y_list = df['stopping_power'].tolist()

#make a cubic spline function
stopping_power = CubicSpline(x_list, y_list)


f = lambda x: 1 / stopping_power(x)

energies = np.linspace(0,7,15)
alpha_ranges = []
for index, energy in enumerate(energies):
    alpha_ranges.append(quad(f, 0, energy)[0])
    print(f"energy: {energy}")
    print(f"range: {alpha_ranges[index]}")

