
from dataclasses import dataclass
from typing import Optional

import pulp as pu
import numpy as np
import matplotlib.pyplot as plt

from .utils_optim import applyMcCormickRelaxation, applyTightMcCormickRelaxation
from .utils_optim import createVariables, _getVarValue

@dataclass
class BatteryWithDegradation(object):
    """Degradation of the Battery

    Child class of the Battery Energy Storage System class

    Attributes
    ----------
    name: str
        Name of battery (used for plotting and debug)
    maximum_charging_power: array-like
        Maximum charging power profile in kW
    maximum_discharging_power: array-like
        Maximum discharging power profile in kW
    state_of_charge: float
        Current SoC, used as initial point for optimization
    capacity : array-like
        Maximum remaining energy capacity profile in kWh, can represent the remaining maximal capacity of the aged battery
        or be used to enforce soC constraints 
    min_capacity : array-like
        Minimum energy capacity profile in kWh, can represent capacity or be used to enforce soC constraints
    max_soc : array-like
        Maximum SoC profile in %, used to enforce SoC constraints
    charging_efficiency : array-like
        One way charging/discharging efficiency. roundtrip efficiency is the square of this value. 
    input_node: str or Node
        Input node
    output_node: str or Node
        Output node. Usually same as input Node
    current_power : float,optional
        Current net power in kW (>=0 for production/discharging, <0 for consumption/charging).
        Used to penalizes change of power
    soc_final_price : float, optional
        Value of SoC at the end of horizon in CHF/kWh.
        tuning parameter for optimization to avoid discharging at the end of the optimization
    soc_avg_price : float, optional
        Value of average SoC over horizon in CHF/kWh.
        Tuning parameter for optimization to promote higher SoC rather than lower
    cycle_cost : float, optional
        Cost of a round trip cycle in CHF/kWh. Tuning parameter for optimization to limit unnecessary cycling
    lambda_soft : float, optional
        Soft constraints cost parameter
    final_soc_constraint : bool, optional
        if True, make sure SoC does not finish lower than at start
    min_terminal_soc: float, optional
        Min SoC at the end of horizon. Only used if final_soc_constraint is True. If None and 
        final_soc_constraint is True, assume to be the initial state of charge
    rate_cost : float
        Cost of power level changes. Tuning parameter for optimization to limit large power variations
    use_linear_model: bool
        If True, does not use binary variable. The optimization problem will result in an exact relaxation
        under conditions of convexity of the cost. Indeed it will be allowed to charge and discharge simultaneously 
        but with a efficiency < 1, this should not happen at the optimum. Use at your own risk.
    battery_cost : float
        Cost of the battery pack, used when the battery needs to be replaced at its End-of-Life
    cell_chemistry : string
        Chemistry of the cells composing the battery: changes its degradation properties
    initial_soh : float
        Initial State of Health (SoH) [%] at the beginning of the optimization, set to 100% if the battery is new
    battery_eol_soh : percentage
        SoH [%] at which the battery is at its End-of-Life
    Mc_for_soc: boolean
        True if we use bilinear constraints relaxed with McCormick enveloppes, False if not. 
    Mc_for_dod : boolean
        True if want to multiply the cycle ageing by the SF for (maximal) DoD over the Horizon,
        False if not (constant value of SF = 1.6 for DoD = 100%). 
    N_Mc: integer
        Number of sub-intervals between the Upper and Lower bounds on each variable of bilinear
        products relaxed with McCormick envelopes. 
    sor_increase : boolean
        Set to False if we ignore the SoR increase in the battery, True if we include it in the optimization.
    battery_eol_sor : percentage
        SoR [%] at which the battery is at its End-of-Life
    """ 
    
    # # Set those variable as tables (even if only a scalar) and gives them vector properties (and set default value to 0)
    # maximum_charging_power    = VectorProperty(0)
    # maximum_discharging_power = VectorProperty(0)
    # capacity                  = VectorProperty(0)
    # charging_efficiency       = VectorProperty(0)
    # min_capacity              = VectorProperty(0)
    # n_input_plots             = 2
    # n_output_plots            = 0

    name: str
    maximum_charging_power : float    
    maximum_discharging_power : float 
    state_of_charge : float            # initial SoC
    capacity : float
    charging_efficiency: float
    # input_node
    # output_node
    # current_power        = None
    soc_final_price : Optional[float]      = 0
    # soc_avg_price        = 0
    cycle_cost : Optional[float]           = 0
    lambda_soft : Optional[float]          = 100  #TODO: could be that?
    final_soc_constraint: Optional[bool]   = False  # if True make sure SoC does not finish lower than at start
    min_terminal_soc : Optional[float]     = None
    rate_cost : Optional[float]            = 0
    min_capacity : Optional[float]         = 0
    max_soc : Optional[float]              = 100 
    min_soc : Optional[float]              = 0
    # use_linear_model     = False
    battery_cost : Optional[float]         = 1_000
    cell_chemistry : Optional[str]         = 'NMC'   
    initial_soh : Optional[float]          = 100          
    battery_eol_soh : Optional[float]      = 80     
    Mc_for_dod : Optional[bool]            = False
    Mc_for_soc : Optional[bool]            = False 
    N_Mc : Optional[float]                 = 3
    sor_increase : Optional[bool]          = False  
    initial_sor : Optional[float]          = 100
    battery_eol_sor : Optional[float]      = 200
        # **kwargs
    # ):
        # super().__init__(                              # ??? no 'name' ? (needed in class Device)
        #     maximum_charging_power,    
        #     maximum_discharging_power, 
        #     state_of_charge,            # initial SoC
        #     capacity,
        #     charging_efficiency,
        #     input_node           = input_node,
        #     output_node          = output_node,
        #     current_power        = current_power,
        #     soc_final_price      = soc_final_price,
        #     soc_avg_price        = soc_avg_price,
        #     cycle_cost           = cycle_cost,
        #     lambda_soft          = lambda_soft,
        #     final_soc_constraint = final_soc_constraint,  # if True, make sure SoC does not finish lower than at start
        #     min_terminal_soc     = min_terminal_soc,
        #     rate_cost            = rate_cost,
        #     min_capacity         = min_capacity,
        #     max_soc              = max_soc,
        #     use_linear_model     = use_linear_model,
        #     **kwargs
        # )

        # # Input parameters
        # self.battery_cost = battery_cost
        # self.cell_chemistry = cell_chemistry
        # self.initial_soh = initial_soh
        # self.battery_eol_soh = battery_eol_soh
        # self.Mc_for_dod = Mc_for_dod
        # self.Mc_for_soc = Mc_for_soc
        # self.N_Mc = N_Mc
        # self.sor_increase = sor_increase
        # self.initial_sor = initial_sor
        # self.battery_eol_sor = battery_eol_sor
        
                
    def __post_init__(self):    
        # Optimization variables from parent's class
        self._variable_definition = [
            ('net_power', None, None, 'continuous'),
            ('pos_power', 0, None, 'continuous'),
            ('neg_power', 0, None, 'continuous'),
            ('state_of_charge', None, None, 'continuous'),
            ('eps', 0, None, 'continuous'),
            ('eps_rate', 0, None, 'continuous'),
            ('char_disc', 0, 1, 'int')
        ]
    
        self._variable_definition.append(('SoC_perc', None, None, 'continuous'))
        self._variable_definition.append(('SoH', None, None, 'continuous'))        
        
        # For PWA function for DR_Crate (4 pieces) function of PWA_Crate
        self._variable_definition.append(('DR_Crate', 0, None, 'continuous'))
        self._variable_definition.append(('PWA_Crate', None, None, 'continuous'))
        # add 4 continuous variables (d) and 3 binary variables (z) (see Report section 7.1)
        self._variable_definition.append(('d1', 0, 1, 'continuous'))
        self._variable_definition.append(('d2', 0, 1, 'continuous'))
        self._variable_definition.append(('d3', 0, 1, 'continuous'))
        self._variable_definition.append(('d4', 0, 1, 'continuous'))
        self._variable_definition.append(('z1', 0, 1, 'int'))
        self._variable_definition.append(('z2', 0, 1, 'int'))
        self._variable_definition.append(('z3', 0, 1, 'int'))
        
        if self.Mc_for_dod:
            # For PWA function for SF_maxDoD (2 pieces) function of max_DoD
            self._variable_definition.append(('SF_maxDoD', 0, None, 'continuous', 1))
            self._variable_definition.append(('max_DoD', 0, 100, 'continuous', 1))
            self._variable_definition.append(('SoC_inf', 0, 100, 'continuous', 1))
            self._variable_definition.append(('SoC_sup', 0, 100, 'continuous', 1))
            # add 2 continuous variables (m) and 1 binary variables (n) (see Report section 7.1)
            self._variable_definition.append(('m1', 0, 1, 'continuous', 1))
            self._variable_definition.append(('m2', 0, 1, 'continuous', 1))
            self._variable_definition.append(('n1', 0, 1, 'int', 1))
            # add DR_Crate_tot (multiplied by SF_maxDoD) relaxed with w_DR_maxDoD (see Report section 7.2 and 7.3)
            self._variable_definition.append(('DR_Crate_tot', 0, None, 'continuous', 1))
            self._variable_definition.append(('w_DR_maxDoD', 0, None, 'continuous', 1))
            
            # For Piecewise Relaxation of N_Mc pieces (see Report section 7.3)
            if self.N_Mc > 1:
                # need (N_Mc-1) binary (int) variables: see Report equation 79
                self._variable_definition.append(('t_DoD', 0, 1, 'int', 1, self.N_Mc -1))
                # need (N_Mc-1) continuous variables 
                self._variable_definition.append(('Dv_DoD', 0, None, 'continuous', 1, self.N_Mc -1))
                # need 2 global incremental variables
                self._variable_definition.append(('Dx_DoD', 0, None, 'continuous', 1))
                self._variable_definition.append(('Dz_DoD', 0, None, 'continuous', 1))
        
        if self.Mc_for_soc:
            # add SF_SoC (multiplied by DR_Crate) relaxed with w_DR (see Report section 7.2 & 7.3)
            self._variable_definition.append(('SF_SoC', 0, None, 'continuous'))
            self._variable_definition.append(('w_DR', 0, None, 'continuous'))
            
            # For Piecewise Relaxation of N_Mc pieces (see Report sections 7.2 & 7.3)
            if self.N_Mc > 1:
                # need (N_Mc-1) binary (int) variables: see Report equation 79
                self._variable_definition.append(('t_SoC', 0, 1, 'int', -1, 4*self.N_Mc -1))
                # need (N_Mc-1) continuous variables 
                self._variable_definition.append(('Dv_SoC', 0, None, 'continuous', -1, 4*self.N_Mc -1))
                # need 2 global incremental variables
                self._variable_definition.append(('Dx_SoC', 0, None, 'continuous'))
                self._variable_definition.append(('Dz_SoC', 0, None, 'continuous'))
            
        if self.sor_increase:
            self._variable_definition.append(('SoR', 100, None, 'continuous'))
            # PWA function for DR_Crate_SoR (5 pieces): add 5 continuous variables (r) and 4 binary variables (b)
            self._variable_definition.append(('DR_Crate_SoR', 0, None, 'continuous'))
            self._variable_definition.append(('r1', 0, 1, 'continuous'))
            self._variable_definition.append(('r2', 0, 1, 'continuous'))
            self._variable_definition.append(('r3', 0, 1, 'continuous'))
            self._variable_definition.append(('r4', 0, 1, 'continuous'))
            self._variable_definition.append(('r5', 0, 1, 'continuous'))
            self._variable_definition.append(('b1', 0, 1, 'int'))
            self._variable_definition.append(('b2', 0, 1, 'int'))
            self._variable_definition.append(('b3', 0, 1, 'int'))
            self._variable_definition.append(('b4', 0, 1, 'int'))
            
            # PWA function for SF_SoC_SoR (3 pieces): add 3 continuous variables (s) and 2 binary variables (c)
            self._variable_definition.append(('SF_SoC_SoR', 0, None, 'continuous'))
            self._variable_definition.append(('s1', 0, 1, 'continuous'))
            self._variable_definition.append(('s2', 0, 1, 'continuous'))
            self._variable_definition.append(('s3', 0, 1, 'continuous'))
            self._variable_definition.append(('c1', 0, 1, 'int'))
            self._variable_definition.append(('c2', 0, 1, 'int'))
            # For McCormick Relaxation of DR_Crate_SoR * SF_SoC_SoR:
            self._variable_definition.append(('w_DR_SoR', 0, None, 'continuous'))
            if self.N_Mc > 1:
                self._variable_definition.append(('t_SoC_SoR', 0, 1, 'int', -1, 3*self.N_Mc -1)) # TODO: check if different from t_SoC
                self._variable_definition.append(('Dv_SoC_SoR', 0, None, 'continuous', -1, 3*self.N_Mc -1))
                self._variable_definition.append(('Dx_SoC_SoR', 0, None, 'continuous'))
                self._variable_definition.append(('Dz_SoC_SoR', 0, None, 'continuous'))
                
            # PWA function for SF_maxDoD_SoR (same 2 pieces as SF_maxDoD: use 2 continuous (m) and 1 binary (n))
            self._variable_definition.append(('SF_maxDoD_SoR', 0, None, 'continuous', 1))
            # add DR_Crate_tot_SoR (multiplied by SF_maxDoD_SoR) relaxed with w_DR_maxDoD_SoR (see Report section 7.2 and 7.3)
            self._variable_definition.append(('DR_Crate_tot_SoR', 0, None, 'continuous', 1))
            self._variable_definition.append(('w_DR_maxDoD_SoR', 0, None, 'continuous', 1))

            # For Piecewise Relaxation of N_Mc pieces (see Report sections 7.2 & 7.3)
            if self.N_Mc > 1:
                # need (N_Mc-1) binary (int) variables: see Report equation 79
                self._variable_definition.append(('t_DoD_SoR', 0, 1, 'int', 1, 3*self.N_Mc -1))
                # need (N_Mc-1) continuous variables 
                self._variable_definition.append(('Dv_DoD_SoR', 0, None, 'continuous', 1, 3*self.N_Mc -1))
                # need 2 global incremental variables
                self._variable_definition.append(('Dx_DoD_SoR', 0, None, 'continuous', 1))
                self._variable_definition.append(('Dz_DoD_SoR', 0, None, 'continuous', 1))


    
    def createVariables(
            self, 
            horizon: int, 
            timestep:int
        ):
        list_variables = self._variable_definition
        self._optimization_horizon = horizon
        self._optimization_timestep = timestep
        self.variables = createVariables(list_variables, horizon, timestep)



    def setModelConstraints(
            self, 
            problem : pu.LpProblem, 
            with_soft : Optional[bool] = False
            ):
        """Constraints for Battery

        Parameters
        ----------
        problem: pu.LpProblem
            Optimization problem to store constraints into
        with_soft: bool
            If True, use soft constraints for SoC limits. this ensures the problem is always feasible
        """
        _K = self._optimization_horizon
        for k in range(_K):
            problem += self.variables['pos_power'][k] - self.variables['neg_power'][k] == self.variables['net_power'][
                k]  # constraint relaxation
        
            problem += self.variables['pos_power'][k] <= self.maximum_discharging_power * (
                1 - self.variables['char_disc'][k])
            problem += self.variables['neg_power'][k] <= self.maximum_charging_power * self.variables['char_disc'][k]
            if with_soft:
                problem += self.variables['state_of_charge'][k] <= self.max_soc / 100 * self.capacity + self.variables['eps'][k]
                problem += self.variables['state_of_charge'][k] >= self.min_capacity - self.variables['eps'][k]
            else:
                problem += self.variables['state_of_charge'][k] >= self.min_capacity
                problem += self.variables['state_of_charge'][k] <= self.max_soc / 100 * self.capacity
            if k == 0:
                problem += self.variables['state_of_charge'][k] == (
                    self.state_of_charge 
                    - self.variables['pos_power'][k]*(1/self.charging_efficiency)*self._optimization_timestep / 60
                    + self.variables['neg_power'][k]*self.charging_efficiency*self._optimization_timestep / 60
                )
            else:
                problem += self.variables['state_of_charge'][k] == (
                    self.variables['state_of_charge'][k-1]
                    - self.variables['pos_power'][k]*(1/self.charging_efficiency)*self._optimization_timestep / 60
                    + self.variables['neg_power'][k]*self.charging_efficiency*self._optimization_timestep / 60
                ) 


            if self.final_soc_constraint:
                final_soc = self.min_terminal_soc if self.min_terminal_soc is not None else self.state_of_charge
                problem += self.variables['state_of_charge'][-1] >= final_soc
        
        # Obtain the Degradation Parameters' data from the cell chemistry
        data = DegradationParameters.from_id(self.cell_chemistry)
        
        # Define the C-rate to get the upper bound on DR_Crate
        maxCrate = self.maximum_charging_power/self.capacity
        # the DR corresponds to the total degradation rate explained in section 5.3.1 (depending on the Crate only, calendar ageing negligible for high Crates)
        maxDR = data.DR_cyc_ref * data.SF_cyc_Cr_D * np.exp(data.SF_cyc_Cr_E * maxCrate) * maxCrate/2
        
        # Common constraints to every model
        for k in range(_K):   
            # Define the SoC in percent [%]
            problem += self.variables['SoC_perc'][k] == 100*self.variables['state_of_charge'][k]/self.capacity
            
            # Defining the Crate from the neg and pos power (including binary variable for Charge/Disch), with efficiency
            problem += self.variables['PWA_Crate'][k] ==(
                    - self.variables['neg_power'][k] * 1 / self.capacity * self.charging_efficiency
                    + self.variables['pos_power'][k] * 1 / self.capacity * (1/self.charging_efficiency)
                ) #1 / self.capacity * self.variables['net_power'][k]
            
            # Inequalities for PWA for DR_Crate (continuous and binary variables): see Report equation 69
            problem +=                       0 <= self.variables['d4'][k]
            problem += self.variables['d4'][k] <= self.variables['z3'][k]
            problem += self.variables['z3'][k] <= self.variables['d3'][k]
            problem += self.variables['d3'][k] <= self.variables['z2'][k]
            problem += self.variables['z2'][k] <= self.variables['d2'][k]
            problem += self.variables['d2'][k] <= self.variables['z1'][k]
            problem += self.variables['z1'][k] <= self.variables['d1'][k]
            problem += self.variables['d1'][k] <= 1
            
            # Define PWA function for DR_Crate: x-axis sub-intervals (see Report equation 67)
            problem += self.variables['PWA_Crate'][k] == (data.xN[0] 
                                                          + (data.xN[1] - data.xN[0]) * self.variables['d1'][k] 
                                                          + (data.xN[2] - data.xN[1]) * self.variables['d2'][k] 
                                                          + (data.xN[3] - data.xN[2]) * self.variables['d3'][k] 
                                                          + (data.xN[4] - data.xN[3]) * self.variables['d4'][k])
            
            # PWA function for the DR_Crate: y-axis sub-intervals (see Report equation 68)
            problem += self.variables['DR_Crate'][k] == (data.yN[0] 
                                                         + (data.yN[1] - data.yN[0]) * self.variables['d1'][k] 
                                                         + (data.yN[2] - data.yN[1]) * self.variables['d2'][k] 
                                                         + (data.yN[3] - data.yN[2]) * self.variables['d3'][k] 
                                                         + (data.yN[4] - data.yN[3]) * self.variables['d4'][k])
            # Get the maximal SoC range (to find maxDoD)
            if self.Mc_for_dod:
                # Find inf & sup limits of SoC, for max_DoD
                problem += self.variables['SoC_inf'][0] <= self.variables['SoC_perc'][k]
                problem += self.variables['SoC_sup'][0] >= self.variables['SoC_perc'][k]
            
        
        if self.Mc_for_dod:
            
            # Compute the maximal DoD with the maximal SoC range (see Report equation 57)
            problem += self.variables['max_DoD'][0] == (
                self.variables['SoC_sup'][0] - self.variables['SoC_inf'][0])
            
            # PWA function for SF_maxDoD: y-axis intervals (see Report equation 68)
            problem += self.variables['SF_maxDoD'][0] == (data.SF_DoD[0] 
                                                          + (data.SF_DoD[1] - data.SF_DoD[0]) * self.variables['m1'][0] 
                                                          + (data.SF_DoD[2] - data.SF_DoD[1]) * self.variables['m2'][0])
            
            # PWA function for SF_maxDoD: x-axis intervals (see Report equation 67)
            problem += self.variables['max_DoD'][0] == (data.DoD[0] 
                                                        + (data.DoD[1] - data.DoD[0]) * self.variables['m1'][0] 
                                                        + (data.DoD[2] - data.DoD[1]) * self.variables['m2'][0])
            
            # Inequalities for PWA for SF_maxDoD_SoR (continuous and binary variables) (see Report equation 69)
            problem +=                       0 <= self.variables['m2'][0]
            problem += self.variables['m2'][0] <= self.variables['n1'][0]
            problem += self.variables['n1'][0] <= self.variables['m1'][0]
            problem += self.variables['m1'][0] <= 1
            
            # Introduce McCormick Relaxation (see Report section 7.2)
            # w_DR_maxDoD = x * y = DR_Crate_tot * SF_maxDoD 
            if self.N_Mc == 1 : # see Report section 7.2
                applyMcCormickRelaxation(problem, 
                                              x  = self.variables['DR_Crate_tot'][0],
                                              xL = 0, 
                                              xU = maxDR * self._optimization_horizon /2 ,
                                              y  = self.variables['SF_maxDoD'][0], 
                                              yL = min(data.SF_DoD), 
                                              yU = max(data.SF_DoD), 
                                              z  = self.variables['w_DR_maxDoD'][0])
            else: # see Report section 7.3
                applyTightMcCormickRelaxation(problem, 
                                                   N_Mc = self.N_Mc, 
                                                   x  = self.variables['DR_Crate_tot'][0],
                                                   xL = 0, 
                                                   xU = maxDR * self._optimization_horizon /2 , 
                                                   y  = self.variables['SF_maxDoD'][0], 
                                                   yL = min(data.SF_DoD), 
                                                   yU = max(data.SF_DoD), 
                                                   z  = self.variables['w_DR_maxDoD'][0], 
                                                   t  = self.variables['t_DoD'][0], 
                                                   Dv = self.variables['Dv_DoD'][0], 
                                                   Dx = self.variables['Dx_DoD'][0], 
                                                   Dz = self.variables['Dz_DoD'][0] )
            
            
            
        
        
        #________________________________________________
        # SF_SoC multiplies cycle ageing: use McCormick at every timestep
        if self.Mc_for_soc:
            
            for k in range(_K):   
                
                # Define the SF_SoC for cycle ageing
                problem += self.variables['SF_SoC'][k] == data.SF_cyc_SoC_A * self.variables['SoC_perc'][k] + data.SF_cyc_SoC_B
                
                # Add McCormick constraints (for all k)
                if self.N_Mc == 1: # see Report section 7.2
                    applyMcCormickRelaxation(problem, 
                                                  x  = self.variables['DR_Crate'][k],
                                                  xL = 0, 
                                                  xU = maxDR,
                                                  y  = self.variables['SF_SoC'][k], 
                                                  yL = data.SF_cyc_SoC_B , 
                                                  yU = data.SF_cyc_SoC_A * 100 + data.SF_cyc_SoC_B,
                                                  z  = self.variables['w_DR'][k] )
                else: # see Report section 7.3
                    applyTightMcCormickRelaxation(problem, 
                                                       N_Mc = 4*self.N_Mc, 
                                                       x  = self.variables['DR_Crate'][k],
                                                       xL = 0, 
                                                       xU = maxDR,
                                                       y  = self.variables['SF_SoC'][k], 
                                                       yL = data.SF_cyc_SoC_B , 
                                                       yU = data.SF_cyc_SoC_A * 100 + data.SF_cyc_SoC_B,
                                                       z  = self.variables['w_DR'][k], 
                                                       t  = self.variables['t_SoC'][k], 
                                                       Dv = self.variables['Dv_SoC'][k], 
                                                       Dx = self.variables['Dx_SoC'][k], 
                                                       Dz = self.variables['Dz_SoC'][k] )
                
                if k == 0:
                    # SoH degradation: see Report section 5.3.3
                    problem += self.variables['SoH'][k] == (
                        self.initial_soh 
                        - (data.DR_cal_ref * (data.SF_cal_A * self.variables['SoC_perc'][k] + data.SF_cal_B)
                           + self.variables['w_DR'][k] * data.SF_cst_DoD) 
                          * self._optimization_timestep / 60
                    )
                    
                # Add the contribution of the Max_DoD (over horizon) at the last timestep 
                elif k == _K-1 and self.Mc_for_dod:
                    
                    # x variable (for McCormick on maxDoD): Sum over the horizon of the DR(C-rate) 
                    problem += self.variables['DR_Crate_tot'][0] == sum(self.variables['w_DR'])
                        
                    # SoH degradation: see Report section 5.3.3 (equation 58)
                    problem += self.variables['SoH'][k] == (
                        self.initial_soh 
                        - (data.DR_cal_ref * (data.SF_cal_A * sum(self.variables['SoC_perc']) + _K * data.SF_cal_B)
                           + self.variables['w_DR_maxDoD'][0]) 
                          * self._optimization_timestep / 60
                    )
                        
                else:
                    # SoH degradation: see Report section 5.3.3
                    problem += self.variables['SoH'][k] == (
                        self.variables['SoH'][k-1]
                        - (data.DR_cal_ref * (data.SF_cal_A * self.variables['SoC_perc'][k] + data.SF_cal_B)
                           + self.variables['w_DR'][k] * data.SF_cst_DoD) 
                          * self._optimization_timestep / 60
                    )
                    
                    
            
        #________________________________________________
        # SoC decoupled from cycle ageing (Mc_for_soc == False)
        else: 
            
            for k in range(_K):   
                
                # Degradation constraints  
                if k == 0:
                    # Constraint for the SoH (at the first timestep): substract calendar and cycle ageing (see Report section 5.3.3, equation 56)
                    problem += self.variables['SoH'][k] == (
                        self.initial_soh 
                        - (data.DR_cal_ref * (data.SF_cal_A * self.variables['SoC_perc'][k] + data.SF_cal_B)
                           + self.variables['DR_Crate'][k] * data.SF_max_cst_DoD) 
                          * self._optimization_timestep / 60
                    )
                    
                elif k == _K-1 and self.Mc_for_dod:
                    
                    # x variable (for McCormick on maxDoD): Sum over the horizon of the DR(C-rate) 
                    problem += self.variables['DR_Crate_tot'][0] == sum(self.variables['DR_Crate']) 
                    
                    # Constraint for the SoH: substract calendar and cycle ageing (see Report section 5.3.3)
                    problem += self.variables['SoH'][k] == (    
                        self.initial_soh 
                        - (data.DR_cal_ref * (data.SF_cal_A * sum(self.variables['SoC_perc']) + _K * data.SF_cal_B)
                            + self.variables['w_DR_maxDoD'][0]) 
                          * self._optimization_timestep / 60
                    )
                    
                else: 
                    # Constraint for the SoH: substract calendar and cycle ageing (see Report section 5.3.3, equation 56)
                    problem += self.variables['SoH'][k] == (
                        self.variables['SoH'][k-1] 
                        - (data.DR_cal_ref * (data.SF_cal_A * self.variables['SoC_perc'][k] + data.SF_cal_B)
                           + self.variables['DR_Crate'][k] * data.SF_max_cst_DoD)
                          * self._optimization_timestep / 60
                    )
                    
                    
                    
                    
        #______________________________________________________________________
        
        if self.sor_increase:
            
            # Common constraints to every model
            for k in range(_K):   
                
                # PWA function for the DR f(C-rate): y-axis intervals (DR) (see Report equation 71)
                problem += self.variables['DR_Crate_SoR'][k] == (data.yR[0] 
                                                                 + (data.yR[1] - data.yR[0]) * self.variables['r1'][k] 
                                                                 + (data.yR[2] - data.yR[1]) * self.variables['r2'][k] 
                                                                 + (data.yR[3] - data.yR[2]) * self.variables['r3'][k] 
                                                                 + (data.yR[4] - data.yR[3]) * self.variables['r4'][k]
                                                                 + (data.yR[5] - data.yR[4]) * self.variables['r5'][k])
                
                # PWA function for the DR f(C-rate): x-axis intervals (C-rate) (see Report equation 70)
                problem += self.variables['PWA_Crate'][k] == (data.xR[0] 
                                                              + (data.xR[1] - data.xR[0]) * self.variables['r1'][k] 
                                                              + (data.xR[2] - data.xR[1]) * self.variables['r2'][k] 
                                                              + (data.xR[3] - data.xR[2]) * self.variables['r3'][k] 
                                                              + (data.xR[4] - data.xR[3]) * self.variables['r4'][k] 
                                                              + (data.xR[5] - data.xR[4]) * self.variables['r5'][k])
                
                # Inequalities for PWA for DR_Crate_SoR (continuous and binary variables) (see Report equation 72)
                problem +=                       0 <= self.variables['r5'][k]
                problem += self.variables['r5'][k] <= self.variables['b4'][k]
                problem += self.variables['b4'][k] <= self.variables['r4'][k]
                problem += self.variables['r4'][k] <= self.variables['b3'][k]
                problem += self.variables['b3'][k] <= self.variables['r3'][k]
                problem += self.variables['r3'][k] <= self.variables['b2'][k]
                problem += self.variables['b2'][k] <= self.variables['r2'][k]
                problem += self.variables['r2'][k] <= self.variables['b1'][k]
                problem += self.variables['b1'][k] <= self.variables['r1'][k]
                problem += self.variables['r1'][k] <= 1
                
                # PWA function for SF_SoC_SoR: y-axis intervals (see Report equation 71)
                problem += self.variables['SF_SoC_SoR'][k] == (data.SF_SoC[0] 
                                                               + (data.SF_SoC[1] - data.SF_SoC[0]) * self.variables['s1'][k] 
                                                               + (data.SF_SoC[2] - data.SF_SoC[1]) * self.variables['s2'][k] 
                                                               + (data.SF_SoC[3] - data.SF_SoC[2]) * self.variables['s3'][k])
                                                               
                # PWA function for SF_SoC_SoR: x-axis intervals (see Report equation 70)
                problem += self.variables['SoC_perc'][k] == (data.SoC[0] 
                                                             + (data.SoC[1] - data.SoC[0]) * self.variables['s1'][k] 
                                                             + (data.SoC[2] - data.SoC[1]) * self.variables['s2'][k] 
                                                             + (data.SoC[3] - data.SoC[2]) * self.variables['s3'][k])
                                                             
                # Inequalities for PWA for SF_SoC_SoR (continuous and binary variables) (see Report equation 72)
                problem +=                       0 <= self.variables['s3'][k]
                problem += self.variables['s3'][k] <= self.variables['c2'][k]
                problem += self.variables['c2'][k] <= self.variables['s2'][k]
                problem += self.variables['s2'][k] <= self.variables['c1'][k]
                problem += self.variables['c1'][k] <= self.variables['s1'][k]
                problem += self.variables['s1'][k] <= 1
                
                # From chemistry dependent data
                maxDR_SoR = data.DR_sor_ref * data.SF_SoR_Cr_D * np.exp(data.SF_SoR_Cr_E * maxCrate) * maxCrate/2
                
                # Add McCormick constraints (for all k)
                if self.N_Mc == 1: # see Report section 7.2
                    applyMcCormickRelaxation(problem, 
                                                  x  = self.variables['DR_Crate_SoR'][k],
                                                  xL = 0, 
                                                  xU = maxDR_SoR, 
                                                  y  = self.variables['SF_SoC_SoR'][k], 
                                                  yL = min(data.SF_SoC), 
                                                  yU = max(data.SF_SoC), 
                                                  z  = self.variables['w_DR_SoR'][k] )
                else: # see Report section 7.3
                    applyTightMcCormickRelaxation(problem, 
                                                       N_Mc = 3*self.N_Mc, 
                                                       x  = self.variables['DR_Crate_SoR'][k],
                                                       xL = 0, 
                                                       xU = maxDR_SoR,  
                                                       y  = self.variables['SF_SoC_SoR'][k], 
                                                       yL = min(data.SF_SoC), 
                                                       yU = max(data.SF_SoC), 
                                                       z  = self.variables['w_DR_SoR'][k], 
                                                       t  = self.variables['t_SoC_SoR'][k], 
                                                       Dv = self.variables['Dv_SoC_SoR'][k], 
                                                       Dx = self.variables['Dx_SoC_SoR'][k], 
                                                       Dz = self.variables['Dz_SoC_SoR'][k] )
                if k == 0:
                    # SoR degradation (see Report section 6.3)
                    problem += self.variables['SoR'][k] == (
                        self.initial_sor + 
                        self.variables['w_DR_SoR'][k] * data.SF_SoR_cst_DoD * self._optimization_timestep / 60
                    )
                    
                # Add the contribution of the Max_DoD (over horizon) at the last timestep 
                elif k == _K-1:
                    
                    # SoH degradation (see Report section 6.3: equation 66)
                    problem += self.variables['SoR'][k] == (
                        self.initial_sor + 
                        self.variables['w_DR_maxDoD_SoR'][0] * self._optimization_timestep / 60
                    )
                       
                else:
                    # SoR degradation (see Report section 6.3)
                    problem += self.variables['SoR'][k] == (
                        self.variables['SoR'][k-1] + 
                        self.variables['w_DR_SoR'][k] * data.SF_SoR_cst_DoD * self._optimization_timestep / 60
                    )
                    
            
            
            # _________ Only 1x over horizon ________________  
            
            # Sum of the DR_Crate over Horizon (to multiply with SF_maxDoD_SoR)
            problem += self.variables['DR_Crate_tot_SoR'][0] == sum(self.variables['w_DR_SoR'])
            
            # PWA function for SF_maxDoD: y-axis intervals (see Report equation 68)
            problem += self.variables['SF_maxDoD_SoR'][0] == (data.SF_SoR_DoD[0] 
                                                              + (data.SF_SoR_DoD[1] - data.SF_SoR_DoD[0]) * self.variables['m1'][0] 
                                                              + (data.SF_SoR_DoD[2] - data.SF_SoR_DoD[1]) * self.variables['m2'][0])
            
            # Introduce McCormick Relaxation
            # w_DR_maxDoD_SoR = x * y = DR_Crate_tot_SoR * SF_maxDoD _SoR
            if self.N_Mc == 1 : # see Report section 7.2
                applyMcCormickRelaxation(problem, 
                                              x  = self.variables['DR_Crate_tot_SoR'][0],
                                              xL = 0, 
                                              xU = maxDR_SoR * self._optimization_horizon /10, 
                                              y  = self.variables['SF_maxDoD_SoR'][0], 
                                              yL = min(data.SF_SoR_DoD), 
                                              yU = max(data.SF_SoR_DoD), 
                                              z  = self.variables['w_DR_maxDoD_SoR'][0])
            else: # see Report section 7.3
                applyTightMcCormickRelaxation(problem, 
                                                    N_Mc = 3*self.N_Mc, 
                                                    x  = self.variables['DR_Crate_tot_SoR'][0],
                                                    xL = 0, 
                                                    xU = maxDR_SoR * self._optimization_horizon /10, 
                                                    y  = self.variables['SF_maxDoD_SoR'][0], 
                                                    yL = min(data.SF_SoR_DoD), 
                                                    yU = max(data.SF_SoR_DoD), 
                                                    z  = self.variables['w_DR_maxDoD_SoR'][0], 
                                                    t  = self.variables['t_DoD_SoR'][0], 
                                                    Dv = self.variables['Dv_DoD_SoR'][0], 
                                                    Dx = self.variables['Dx_DoD_SoR'][0], 
                                                    Dz = self.variables['Dz_DoD_SoR'][0] )

    def defineObjective(self, variable_data, with_soft=False):
        """Cost contribution for Battery

        Includes rate cost, charging cost, final SoC value and Average SoC value
        """
        objective = {}

        # model cost
        # First component: terminal cost of SoC
        objective['soc_terminal_price'] = -self.soc_final_price * variable_data['state_of_charge'][-1]

        # Rate cost
        objective['rate_cost'] = 0
        if self.rate_cost != 0:
            for k in range(self._optimization_horizon):
                objective['rate_cost'] += (
                    self.rate_cost * variable_data['eps_rate'][k] if variable_data['eps_rate'][k] is not None else 0
                )
        # Soft constraints
        if with_soft:
            objective['penalty_violation'] = 0
            for k in range(self._optimization_horizon):
                objective['penalty_violation'] += (
                    variable_data['eps'][k] * self.lambda_soft * self._optimization_timestep / 60
                )
        # Cost of cycling
        objective['cycling_cost'] = 0
        for k in range(self._optimization_horizon):
            objective['cycling_cost'] += (
                self.cycle_cost
                * (self._optimization_timestep / 60) / 2
                * (variable_data['pos_power'][k]*(1/self.charging_efficiency) 
                + variable_data['neg_power'][k]*self.charging_efficiency)
            )
        
        # Degradation cost (see Report equations 35 and 60)
        if self.sor_increase:
            objective['SoR_degrad_cost'] = (self.battery_cost * 0.5 
                                            * (variable_data['SoR'][-1] - self.initial_sor) / (self.battery_eol_sor - 100)
            )
            objective['SoH_degrad_cost'] = (self.battery_cost * 0.5
                                            * (self.initial_soh - variable_data['SoH'][-1]) / (100 - self.battery_eol_soh)
            )
        else: 
            objective['SoH_degrad_cost'] = (self.battery_cost * (self.initial_soh - variable_data['SoH'][-1]) / (100 - self.battery_eol_soh)
            )

        return objective

    def retrieveResultsFromVars(self):
        """Retrieves all variables values

        Compiles results of optimization in result attribute
        """
        self.results = dict()
        #
        for name, var in self.variables.items():
            self.results[name] = _getVarValue(var)
    
    def plotResults(self):
        """Retrieves all variables values

        Compiles results of optimization in result attribute
        """
        fig, ax = plt.subplots(2,1, sharex=True)
        fig.suptitle(self.name)
        ax[0].plot(
            np.arange(self._optimization_horizon) * self._optimization_timestep / 60,
            self.results['state_of_charge'],
            marker='x',
            markersize=4
        )
        ax[0].plot(
            np.arange(self._optimization_horizon) * self._optimization_timestep / 60, 
            self.capacity*np.ones(self._optimization_horizon)
        )
        ax[0].plot(
            np.arange(self._optimization_horizon) * self._optimization_timestep / 60, 
            self.min_capacity*np.ones(self._optimization_horizon)
        )
        ax[0].set_ylabel('Energy [kWh]')
        ax[0].legend(['Optimal charge', 'Capacity', 'Min. capacity'], fontsize='x-small')
        ax[0].xaxis.set_ticklabels([])
        ax[0].grid('on')

        ax[1].plot(
            np.arange(self._optimization_horizon) * self._optimization_timestep / 60,
            -self.results['net_power'],
            marker='x',
            markersize=4
        )
        ax[1].plot(
            np.arange(self._optimization_horizon) * self._optimization_timestep / 60,
            -self.maximum_discharging_power*np.ones(self._optimization_horizon)
        )
        ax[1].plot(
            np.arange(self._optimization_horizon) * self._optimization_timestep / 60,
            self.maximum_charging_power*np.ones(self._optimization_horizon)
        )
        ax[1].set_ylabel('Power [kW]')
        ax[1].legend(['Net power', 'Max. discharging', 'Max. charging'], fontsize='x-small')
        ax[1].xaxis.set_ticklabels([])
        ax[1].grid('on')


@dataclass
class DegradationParameters(object):
    
    #______________ Calendar ageing ________________
    # Reference Degradation Rate (DR)
    DR_cal_ref: float = 0  # [%SoH/min] at 25°C, 50% SoC
    # Coefficients for linear SF equation of State of Charge (SoC)
    SF_cal_A: float = 0
    SF_cal_B: float = 0
    
    #________________ Cycle ageing ________________
    # Reference Degradation Rate (DR)
    DR_cyc_ref: float = 0
    
    # Points of PWA approximation of DR_cyc for SoH: f(C-rate)
    xN: float = 0
    yN: float = 0
    
    # Constant SF for DoD
    SF_max_cst_DoD: float = 0
    SF_cst_DoD: float = 0 
    # Points of PWA approximation of SF_maxDoD
    DoD: float = 0
    SF_DoD: float = 0
    
    # Coefficients for linear SF equation of SoC
    SF_cyc_SoC_A: float = 0
    SF_cyc_SoC_B: float = 0 
    
    # Coefficients for exponential SF equation for C-rate: D*exp(E*Crate)
    SF_cyc_Cr_D: float = 0, 
    SF_cyc_Cr_E: float = 0,
    
    #________________ SoR ________________
    # Reference Degradation Rate (DR)
    DR_sor_ref: float = 0 # [%SoR/FEC] at 25°C, 100% DoD, 50% SoC and 0.33C ch/disch
    
    # Points of PWA approximation of DR_cyc for SoH: f(C-rate)
    xR: float = 0
    yR: float = 0
    
    # Points of PWA approximation of SF_SoC_SoR
    SoC: float = 0
    SF_SoC:float = 0
    
    # Constant SF for DoD
    SF_SoR_cst_DoD: float = 0
    # Points of PWA approximation of SF_maxDoD_SoR
    SF_SoR_DoD: float = 0
    
    # Coefficients for exponential SF equation for C-rate: D*exp(E*Crate)
    SF_SoR_Cr_D:float = 0 
    SF_SoR_Cr_E:float = 0

    @staticmethod
    def from_id(s):
        '''Allows to generate pre-specified degradation parameter lists for various cell chemistries, variations
        '''
        if s=='NMC':
            #These values are based on the SoXery degradation model that was built based on empirical tests
            #The parameters are currently as published on https://gitlab.ti.bfh.ch/oss/esrec/open-sesame 

            return DegradationParameters(
                #________________ Calendar ageing ________________
                # Reference Degradation Rate for Calendar ageing
                DR_cal_ref = 0.003337/24,  # [%SoH/h] at 25°C, 50% SoC
                # Coefficients for linear SF equation of State of Charge (SoC)
                SF_cal_A = 0.0121,
                SF_cal_B = 0.3961, 
                
                #________________ Cycle ageing ________________
                # Reference Degradation Rate for Cycle ageing (only used to find max_DR)
                DR_cyc_ref = 0.000284, # [%SoH/FEC] at 25°C, 50% SoC, 20% DoD, 0.5C Charge, 1C Discharge
                
                # Points of PWA approximation of DR_cyc for SoH: f(C-rate)
                xN = [-1,    -0.75,  -0.5 ,    0, 2],           # C-rate
                yN = [0.002, 0.00035, 0.000064, 0, 0.00032], # DR in [%/h]
                
                # Constant SF for DoD
                SF_max_cst_DoD = 1.7, # SF value for DoD = 100%
                SF_cst_DoD = 1.4, # SF value for DoD = 80%
                DoD    = [0,   85,  100],
                SF_DoD = [0.9, 1.28, 1.7],
                
                # Coefficients for cycle ageing linear SF equation of SoC
                SF_cyc_SoC_A =  0.0076, 
                SF_cyc_SoC_B = 0.65, 
                
                # Coefficients for exponential SF equation for C-rate: D*exp(E*Crate)
                SF_cyc_Cr_D = 0.0625, 
                SF_cyc_Cr_E = 5.546,
                
                #________________ SoR ________________
                # Reference Degradation Rate (DR) (only used to find max_DR))
                DR_sor_ref = 0.015, # [%SoR/FEC] at 25°C, 100% DoD, 50% SoC and 0.33C ch/disch
                
                # Points of PWA approximation of DR_cyc for SoR: f(C-rate)
                xR = [-1,   -0.5,   -0.3 , 0, 1, 2],     # C-rate
                yR = [0.185, 0.008, 0.0015, 0.0, 0.008, 0.03],  # DR in [%/h]
                
                # Points of PWA approximation of SF_SoC_SoR (3 pieces)
                SoC    = [0,  35, 70, 100], 
                SF_SoC = [4.7, 1.1, 1.1, 4], 
                
                # Points of PWA approximation of SF_maxDoD_SoR (2 pieces)
                # DoD      = [0,   60,  100] same as for SoH_cyc
                SF_SoR_DoD = [0.06, 0.35, 1],
                
                # Coefficients for exponential SF equation for C-rate: D*exp(E*Crate)
                SF_SoR_Cr_D = 0.19, 
                SF_SoR_Cr_E = 5.055,
            )
        
        
        
        elif s=='NMC_newOS':
            #These values are based on an upgrade of the SoXery degradation model that was built based on empirical tests
            #We recommend uisng those values 
            return DegradationParameters(
                #________________ Calendar ageing ________________
                # Reference Degradation Rate for Calendar ageing
                DR_cal_ref = 0.003337/24,  # [%SoH/h] at 25°C, 50% SoC
                # Coefficients for linear SF equation of State of Charge (SoC)
                SF_cal_A = 0.0121,
                SF_cal_B = 0.3961, 
                
                #________________ Cycle ageing ________________
                # Reference Degradation Rate for Cycle ageing (Charging, because only used to find max_DR)
                DR_cyc_ref = 0.00932, # [%SoH/HEC] at 45°C, 50% SoC, 60% DoD, 1C Charge, 1C Discharge
                
                # Points of PWA approximation of DR_cyc for SoH: f(C-rate)
                xN = [-1, -0.5, 0, 1, 2 ],           # C-rate
                yN = [0.0052, 0.0015, 0, 0.00055, 0.0018], # DR in [%/h]
                
                # Constant SF for DoD
                SF_max_cst_DoD = 1.6, # SF value for DoD = 100%
                SF_cst_DoD     = 1.1, # SF value for DoD = 70%, when using the option Mc_for_dod (reduces the DoD of cycles)
                # Points of PWA approximation of SF_maxDoD (2 pieces)
                DoD    = [0,   70,  100],
                SF_DoD = [0.75, 1.04, 1.6],
                
                # Coefficients for cycle ageing linear SF equation of SoC
                SF_cyc_SoC_A =  0.0072, 
                SF_cyc_SoC_B = 0.6376, 
                
                # Coefficients for exponential SF equation for charging C-rate: D*exp(E*Crate)
                SF_cyc_Cr_D = 0.3858, 
                SF_cyc_Cr_E = 1.035,
                
                #________________ SoR ________________
                # Reference Degradation Rate (DR) (Charging, because only used to find max_DR))
                DR_sor_ref = 0.0082, # [%SoR/FEC] at 25°C, 100% DoD, 50% SoC and 0.33C ch/disch
                
                # Points of PWA approximation of DR_cyc for SoR: f(C-rate)
                xR = [-1, -0.7,  -0.5, -0.3,  0, 1, 2],     # C-rate
                yR = [0.4, 0.05, 0.01, 0.0017, 0.0, 0.0016, 0.0055],  # DR in [%/h]
                
                # Points of PWA approximation of SF_SoC_SoR (4 pieces)
                SoC    = [0,   30,  55,    75 , 100], 
                SF_SoC = [4.7, 1.5, 1, 1.5, 4], 
                
                # Constant SF for DoD
                SF_SoR_cst_DoD     = 0.4, # SF value for DoD = 70%, when using the option Mc_for_dod (reduces the DoD of cycles)
                # Points of PWA approximation of SF_maxDoD_SoR (2 pieces)
                # DoD      = [0,   70,  100] same as for SoH_cyc
                SF_SoR_DoD = [0.06, 0.4, 1],
                
                # Coefficients for exponential SF equation for C-rate: D*exp(E*Crate)
                SF_SoR_Cr_D = 0.1394, 
                SF_SoR_Cr_E = 5.97,
            )         