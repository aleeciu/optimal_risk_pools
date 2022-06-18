import pandas as pd
import numpy as np

from pymoo.core.problem import ElementwiseProblem

def calc_pool_conc(x, data_arr, bools, alpha):
    """Calculate diversification of a given pool. Used to 
    find the best pool.

    x : bool
        Countries to consider in the pool
    data_arr : np.array
        Numpy array with annual damages for all countries
    bools : np.array
        Numpy array with the same shape as data, indicating when 
        annual damages are higher/lower than the country VaR
    alpha : float
        Point at which to calculate VaR and ES
    """

    dam = data_arr[:,x]
    cntry_bools = bools[:,x]
    tot_damage = dam.sum(1)

    VAR_tot = np.quantile(tot_damage[~np.isnan(tot_damage)], alpha)
    bool_tot = tot_damage >= VAR_tot

    ES_cntry = []
    MES = []

    for cntry_pos in range(dam.shape[1]):
        dummy_dam = dam[:,cntry_pos][cntry_bools[:,cntry_pos]]

        ES_cntry.append(np.nanmean(dummy_dam))
        MES.append(np.nanmean(dam[:,cntry_pos][bool_tot]))

    ES_cntry = np.array(ES_cntry)
    MES = np.array(MES)

    # if no countries are picked
    if x.sum() == 0:
        POOL_CONC = 1.
        IND_CONC = 1.
    else:
        IND_CONC = MES / ES_cntry
        ES_tot = np.nansum(MES)
        POOL_CONC = ES_tot / np.nansum(ES_cntry)

    return np.round(POOL_CONC, 2), IND_CONC, MES, ES_cntry, tot_damage

def calc_pools_conc(x, data, bools, alpha, N, fixed_pools=None):
    """Calculate diversification of N pools where all passed countries
    must be in one pool

    x : np.array
        Integers. Integers assess what pool do countries join. 
        It must hold that x.size equals + fixed_pools.size equals n_countries.
    data : np.array
        Numpy array with annual damages for all countries
    bools : np.array
        Numpy array with the same shape as data, indicating when 
        annual damages are higher/lower than the country VaR
    alpha : float
        Confidence level to estimating VaR and ES
    N : int
        Number of pools
    fixed_pools : np.array
        Integers for countries which will always join the same pool. It 
        concatenates x from the left (beginning of the array). It must hold that 
        x.size equals + fixed_pools.size equals n_countries.
    """

    CONC_POOL = []

    if fixed_pools is not None:
        x = np.hstack([fixed_pools, x])

    for i in range(1, N+1):
        countries_in_pool = x == i
        conc_pool = calc_pool_conc(countries_in_pool, data, bools, alpha)[0]

        CONC_POOL.append(conc_pool)

    return np.array([CONC_POOL])

class MinConcProblem(ElementwiseProblem):
    def __init__(self, data, bools, alpha, fun, **kwargs):
        self.data_arr = data.values
        self.bools = bools
        self.alpha = alpha
        self.fun = fun

        super().__init__(n_var=self.data_arr.shape[1],
                         n_obj=1,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # pool's concentration and individual concentration
        pool_conc = self.fun(x, self.data_arr, self.bools, self.alpha)[0]

        out["F"] = pool_conc

class MinNumCntrProblem(ElementwiseProblem):
    def __init__(self, data, bools, alpha, fun, min_conc, **kwargs):
        self.data_arr = data.values
        self.bools = bools
        self.alpha = alpha
        self.fun = fun
        self.min_conc = min_conc

        super().__init__(n_var=self.data_arr.shape[1],
                         n_obj=1,
                         n_constr=1,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # pool's concentration and individual concentration
        pool_conc = self.fun(x, self.data_arr, self.bools, self.alpha)[0]

        out["F"] = np.sum(x)
        out["G"] = (pool_conc-self.min_conc)

class MinConcsProblem(ElementwiseProblem):
    def __init__(self, n_var, data, bools, alpha, fun, N,
                                    fixed_pools=None, **kwargs):
        self.data_arr = data.values
        self.bools = bools
        self.alpha = alpha
        self.n_var = n_var
        self.fun = fun
        self.N = N
        self.fixed_pools = fixed_pools

        super().__init__(n_var=n_var,
                         n_obj=self.N,
                         xl=0,
                         xu=self.N,          
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # pool's concentration and max individual concentration
        pools_conc = self.fun(x, self.data_arr, self.bools, 
                            self.alpha, self.N, self.fixed_pools)

        out["F"] = pools_conc

class MinNumCntrProblems(ElementwiseProblem):
    def __init__(self, n_var, data, bools, alpha, fun, N, min_conc, fixed_pools=None, **kwargs):
        self.data_arr = data.values
        self.bools = bools
        self.alpha = alpha
        self.n_var = n_var
        self.N = N
        self.fun = fun
        self.min_conc = min_conc
        self.fixed_pools = fixed_pools

        super().__init__(n_var=self.n_var,
                         n_obj=1,
                         n_constr=1,           
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # pool's concentration and individual concentration
        pool_conc = self.fun(x, self.data_arr, self.bools, 
                            self.alpha, self.N, self.fixed_pools)

        out["F"] = np.sum(x)
        out["G"] = (pool_conc-self.min_conc)