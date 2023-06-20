from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pulp as pu

# sys.path.append(os.path.realpath(os.path.join(os.getcwd(), '..')))
# print(sys.path)
# from simulation_input import simulation_input
# from pentagon import bess_degrad_model as degrad_model
# from pentagon import config
# from pentagon import pentagon_models as pm
# from pentagon.util import PreLoadedSensor

# config._K = 96  # horizon in [number of timesteps]
# from pentagon import hand_written_objects as hwo

from src import models
from src.utils_optim import defineLinearCost


# Battery data
cell_chemistry       = 'NMC_newOS'          # 'NMC' or 'NMC_newOS' for the old/new Open Sesame (OS) model
max_charge_Crate     = 1    
max_disch_Crate      = 2
batt_en_capacity     = 5                    # [kWh] (start with 5 kWh in maestro)
min_capacity         = 0                    # [kWh] to enforce SoC constraint
batt_cost_per_kWh    = 200                  # [CHF/kWh] 
initial_SoC          = batt_en_capacity/2   # [%] (start with 2.5 kWh in maestro)
charging_efficiency  = 0.95                 # One way efficiency
battery_eol_sor      = 200                  # [%] Considered end of life for SoR

price_energy = np.concatenate([0.5*np.ones(5), 0.1*np.ones(5), 0.2*np.ones(5), 0.5*np.ones(5), 0.1*np.ones(4)])


# Simulation Set-up
K_step = 60                                 # duration of one timestep in [min]
tph = 60 // K_step                          # number of full timesteps per hour (tph)

N_Mc = 10                                   # Number of sub-intervals between Lower and Upper bounds (for PW McCormick)


def run_optim_batt_soh(
    complex_SoH: Optional[bool] = False, 
    include_SoR: Optional[bool] = False,
    H: Optional[int] = 24
    ):
    '''Run optimization test with battery connected to grid in arbitrage scenario

    Parameters
    ----------
    complex_SoH: bool
        If True, use complex SoH model with PW McCormick relaxation
    include_SoR: bool
        If True, include SoR model in optimization
    H: int
        Horizon of optimization
    '''
    

    bess = models.BatteryWithDegradation(
            name = 'bess',
            maximum_charging_power = max_charge_Crate * batt_en_capacity,          # [kW]
            maximum_discharging_power = max_disch_Crate * batt_en_capacity,        # [kW]
            state_of_charge = initial_SoC/100*batt_en_capacity,                    # initial capacity [kWh]
            capacity = batt_en_capacity,                                           # maximal capacity [kWh]
            min_capacity = min_capacity,                                           # minimal capacity [kWh]
            charging_efficiency = charging_efficiency,
            battery_cost = batt_cost_per_kWh*batt_en_capacity,    # [CHF] 
            cell_chemistry = cell_chemistry, 
            initial_soh = 100,                                    # [%]
            battery_eol_soh = 80,                                 # [%]
            Mc_for_dod = complex_SoH,
            N_Mc = N_Mc, 
            Mc_for_soc = complex_SoH,
            sor_increase = include_SoR, 
            battery_eol_sor = battery_eol_sor,                                 # [%]
            )

    ## Setup optimization problem (relies on library PuLP for linear programming)
    problem = pu.LpProblem('ctrl', pu.LpMinimize)
    bess.createVariables(H, K_step)

    # Creating all constraints on the battery variables
    bess.setModelConstraints(problem, with_soft = True)

    # Defining optimization cost 
    # Cost for arbitrage
    cost_arbitrage = defineLinearCost(bess.variables['neg_power'], price_energy) - defineLinearCost(bess.variables['pos_power'], price_energy)
    # Cosst for battery, including degradation
    cost_battery = bess.defineObjective(bess.variables, with_soft = True)
    problem += cost_arbitrage + sum(cost_battery.values())
    
    # create the optimization problem
    gap = 0.00005
    # Solver: possible to use other solvers
    solver = pu.apis.PULP_CBC_CMD(msg=0, gapRel=gap, timeLimit = 180)
    
    
    ## Solve the problem 
    problem.solve(solver=solver)

    # Collect results for all variables    
    bess.retrieveResultsFromVars()
    # Plot results
    bess.plotResults()
    # Optimal value of objective
    cost = pu.value(problem.objective)
    print('Cost = ', cost)

    # Status of optimization
    status = pu.LpSolution[problem.sol_status]
    print ('Status:', status)
    if status != 'Optimal Solution Found':
        print('Time Limit reached!')
    return cost, gap

def test_batt_simple_soh():
    '''Test battery degradation with simple model of SoH, no SoR'''
    cost, gap = run_optim_batt_soh(True, False)
    # expected_cost = -1.8486
    # assert np.isclose(cost, expected_cost, rtol = gap)

def test_batt_complex_soh():
    '''Test battery degradation with complex model of SoH, no SoR'''
    cost, gap = run_optim_batt_soh(True, True)
    # expected_cost = -1.4932
    # assert np.isclose(cost, expected_cost, rtol = gap)
    
def test_batt_complex_soh_sor():
    '''Test battery degradation with complex model of SoH and of SoR'''
    cost, gap = run_optim_batt_soh(True, True)
    # expected_cost = -1.3039
    # assert np.isclose(cost, expected_cost, rtol = gap)

if __name__ == "__main__":
    
    # test_batt_simple_soh()
    # test_batt_complex_soh()
    test_batt_complex_soh_sor()
    
    plt.show()