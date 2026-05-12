import math
import numpy as np

TURBINE = {
    'rotor_diameter_m': 120,
    'rated_power_kW': 3000,
    'cut_in_ms': 3.0,
    'rated_wind_ms': 12.0,
    'cut_out_ms': 25.0,
    'power_coefficient': 0.48,
}

swept_area = math.pi * (TURBINE['rotor_diameter_m'] / 2) ** 2

def air_density(T_celsius, RH_percent, pressure_hPa):

    T = T_celsius + 273.15

    e_sat = 6.112 * np.exp((17.67 * T_celsius) / (T_celsius + 243.5))

    e = (RH_percent / 100.0) * e_sat

    T_v = T * (1 + 0.608 * (e / pressure_hPa))

    rho = (pressure_hPa * 100) / (287.058 * T_v)

    return rho

def wind_power(ws, rho):

    if ws < TURBINE['cut_in_ms'] or ws > TURBINE['cut_out_ms']:
        return 0.0

    return 0.5 * rho * swept_area * (ws ** 3) * TURBINE['power_coefficient']