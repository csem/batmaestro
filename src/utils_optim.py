from typing import Optional, List

import pulp as pu
import numpy as np

def create2dLpMatrix(
        dim : tuple, 
        name : str, 
        var_type: str, 
        lowBound: Optional[float]=None, 
        upBound: Optional[float]=None
    ):
    '''
    By default, pulps supports only 1d Lp variables
    create2dLpMatrix creates a 2D structure with Lp matrices
    Input:
    - dim : tuple with 2 elements for the dimension
    - name: root name of the variable
    '''
    n,m = dim
    variables_matrix = np.empty(dim, dtype = pu.LpVariable)
    for i in range(n):
        variables = pu.LpVariable.matrix(name = '{}_{}'.format(name, i), 
                                            indexs = (list(range(m))), 
                                            lowBound = lowBound,  
                                            upBound = upBound,
                                            cat = var_type)
        variables_matrix[i] = variables
    return variables_matrix

def createVariables(
    list_variables : List[tuple], 
    horizon: int, 
    timestep: int
    ):  # TODO: this should be private probably, could use utility function
    """Generate variable objects

    Parameters
    ----------
    horizon: int
        Horizon of optimization in timesteps. Determines dimension of variables
    timestep: int
        Time step of optimization in minutes

    Notes
    -----
    The variable definition allows flexibility in how the size of the variable is defined
    , controlled by the length of the tuple that describes the variable
    The first 4 elements are the name of the variable, lower bound, upper bound and type
    If there is 4 elements, then the variable is of size = horizon
    If there is a 5th element, the variable has size (var_info[4])
    If there is a sixth element, then the variable is two-dimensional of size (var_info[4], var_info[5])
        If, with six elements, var_info[4] is -1, then put it to the horizon (when available)
    """
    # list_variables = self._variable_definition
    # self._optimization_horizon = horizon
    # self._optimization_timestep = timestep
    variables = dict()
    map_type = {'int': pu.LpInteger, 'continuous': pu.LpContinuous}
    for var_info in list_variables:
        var_name = var_info[0]
        lb = var_info[1]
        ub = var_info[2]
        var_type = var_info[3]
        var = map_type[var_type]
        if len(var_info) == 4:
            var_size_0 = horizon
            var_size_1 = None
        if len(var_info) == 5:
            var_size_0 = var_info[4]
            var_size_1 = None
        if len(var_info) == 6:
            if var_info[4] == -1 :
                var_size_0 = horizon
            else:
                var_size_0 = var_info[4]
            var_size_1 = var_info[5]
        if var_size_1 is None:
            variables[var_name] = pu.LpVariable.matrix(
                name= var_name,
                indexs=(list(range(var_size_0))),
                lowBound=lb,
                upBound=ub,
                cat=var
            )
        else:
            variables[var_name] = create2dLpMatrix(
                name= var_name,
                dim=(var_size_0, var_size_1),
                lowBound=lb,
                upBound=ub,
                var_type=var
            )
    return variables

def defineLinearCost(
    variable: pu.LpVariable,
    cost_vector: List[float]
    ):
    cost = 0
    for k, v in enumerate(variable):
        cost += cost_vector[k] * v
    return cost

def _getVarValue(var):
    if isinstance(var,list):
        return np.array(
            [elt.value() for elt in var]
        )
    elif isinstance(var, np.ndarray):
#        nd = var.ndim
        res = np.zeros(var.shape)
        it = np.nditer(var, flags=['multi_index', 'refs_ok'])
        while not it.finished:
#            print("{} {}".format(it[0], it.multi_index), end=' ')
            res[it.multi_index] = it[0].item().value()
            it.iternext()
        return res
    elif isinstance(var, pu.LpVariable) or isinstance(var, pu.LpAffineExpression):
        res = var.value()
        return res

def applyMcCormickRelaxation(
    problem: pu.LpProblem, 
    x: pu.LpVariable, 
    xL: float, 
    xU: float, 
    y: pu.LpVariable, 
    yL: float, 
    yU: float, 
    z: pu.LpVariable
    ):
    """Constraints for Relaxation of bilinear terms

    Add to the problem the additional constraints from the McCormick Relaxation of bilinear terms z = x*y.
    Details and explanations about the additional variables and constraints 
    can be found in the Master Thesis of Axel Sutter, section "7.2 McCormick Relaxation". 
    
    Inputs: 
        problem
        x : first variable (partitioned)
        xL: lower bound of the first variable
        xU: Upper bound of the first variable
        y : second variable (not partitioned)
        yL: lower bound of the second variable
        yU: Upper bound of the second variable
        z : new variable to relax x*y
        
    Output:
        problem : with the new constraints added
    """
    
    # Normal McCormick constraints (equations 73-76)
    problem += z >= x*yL + xL*y - xL*yL
    problem += z >= x*yU + xU*y - xU*yU
    problem += z <= x*yL + xU*y - xU*yL
    problem += z <= x*yU + xL*y - xL*yU
    

def applyTightMcCormickRelaxation(
    problem: pu.LpProblem, 
    N_Mc: int,
    x: pu.LpVariable, 
    xL: float, 
    xU: float, 
    y: pu.LpVariable, 
    yL: float, 
    yU: float, 
    z: pu.LpVariable, 
    t: pu.LpVariable, 
    Dv: pu.LpVariable, 
    Dx: pu.LpVariable, 
    Dz: pu.LpVariable
    ):
    """Constraints for Tight Piecewise Relaxation of bilinear terms

    Add to the problem the additional constraints from the Tight Piecewise McCormick Relaxation of bilinear terms z = x*y.
    Details and explanations about the additional variables and constraints can be found in the Master Thesis 
    of Axel Sutter, section "7.3 Tight Piecewise McCormick Relaxation". 
    
    Inputs: 
        problem
        N_Mc : Number of sub-interval for the x partitioning of the McCormick relaxation
        x : first variable (partitioned)
        xL: lower bound of the first variable
        xU: Upper bound of the first variable
        y : second variable (not partitioned)
        yL: lower bound of the second variable
        yU: Upper bound of the second variable
        z : new variable to relax x*y
        t : binary variable of size N-1
        Dv: continuous variable of size N-1
        Dx: global incremental variable
        Dz: global incremental variable
        
    Output:
        problem : with the new constraints added
    """
    
    # Length of sub-intervals
    d = (xU - xL)/N_Mc
        
    # Additional constraints from Piecewise McCormick Relaxation (see Report equation 78)
    problem +=  0 <= Dx 
    problem += Dx <= d
    
    problem += Dz <= (yU - yL) * Dx                  # (equation 93)
    problem += Dz <= d * (y - yL)                    # (equation 92)
    problem += Dz >= ((yU - yL) * Dx + d * (y - yU)) # (equation 91)
    
    # Constraints on the N-1 continuous & binary variables (for n = 0,..,N-2)
    
    # First constraints (n = 0)
    problem +=  t[0] >= (x -xL - d) /(xU - xL)       # (equation 80)
    problem += Dv[0] <= y - yL                       # (equation 85)
    problem += Dv[0] >= ((yU - yL) * t[0] + y - yU)  # (equation 88)
    
    for n in range(1, N_Mc - 2):
        
        problem += t[n-1]  >= t[n]                                  # (equation 81)
        problem += t[n]    >= (x -xL - (n+1)*d) /(xU - xL)          # (equation 80)
        problem += Dv[n-1] >= Dv[n]                                 # (equation 86)
        problem += Dv[n]   >= ((yU - yL) *(t[n] -t[n-1]) + Dv[n-1]) # (equation 89)
    
    if N_Mc >= 3:
        # Last constraints (n = N_Mc - 2)
        n = N_Mc - 2
        problem += t[n-1]  >= t[n]                        # (equation 81)
        problem += t[n]    >= (x -xL -(n+1)*d) /(xU - xL) # (equation 80)
        problem += Dv[n-1] >= Dv[n]                       # (equation 86)
        problem += Dv[n]   >= 0                           # (equation 87)
        problem += Dv[n]   <= t[n] * (yU - yL)            # (equation 90)
    
    # Constraint on x (equation 77)
    problem += x == xL + Dx + d * sum(t) 
    # Constraint on y (equation 82)
    problem += y <= yU
    # Constraint on z (equation 84)
    problem += z == (yL * x + xL * y - xL * yL + Dz + d * sum(Dv))
    