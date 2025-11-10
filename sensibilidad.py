# sensibilidad.py

import numpy as np
from diferencias_finitas import ModelParams, GridParams, price_surface_fd
from optimizacion import fd_prices_at_S0V0, make_objective_1param

def sensibilidad_objetivo_1param(TK_list,
                                 P_target: np.ndarray,
                                 params_base: ModelParams,
                                 grid: GridParams,
                                 param_name: str,
                                 bounds: tuple[float, float],
                                 dt_target: float | None,
                                 use_cfl: bool = True,
                                 factor_min: float = 0.7,
                                 factor_max: float = 1.3,
                                 n_points: int = 25):
    """
    Devuelve una malla 1D de valores (param_grid, J_values)
    para el análisis de sensibilidad del funcional J respecto a param_name.
    """
    param_0 = getattr(params_base, param_name)
    J = make_objective_1param(TK_list, P_target, params_base, grid,
                              param_name=param_name,
                              bounds=bounds,
                              dt_target=dt_target,
                              use_cfl=use_cfl)

    param_grid = np.linspace(factor_min * param_0, factor_max * param_0, n_points)
    J_values  = np.empty_like(param_grid, dtype=float)

    for i, val in enumerate(param_grid):
        J_values[i] = J(np.array([val], dtype=float))

    return param_grid, J_values

def sensibilidad_precio_1param(T: float, K: float,
                               params_base: ModelParams,
                               grid: GridParams,
                               param_name: str,
                               bounds: tuple[float, float],
                               dt_target: float | None,
                               use_cfl: bool = True,
                               factor_min: float = 0.7,
                               factor_max: float = 1.3,
                               n_points: int = 25):
    """
    Devuelve (param_grid, prices) para el análisis de sensibilidad del
    precio F_FD(0,S0,V0;T,K) respecto a param_name.
    """
    param_0 = getattr(params_base, param_name)
    param_grid = np.linspace(factor_min * param_0, factor_max * param_0, n_points)
    prices = np.empty_like(param_grid, dtype=float)

    for i, val in enumerate(param_grid):
        val_clip = float(np.clip(val, bounds[0], bounds[1]))
        params = ModelParams(**vars(params_base))
        setattr(params, param_name, val_clip)

        F0, S, V = price_surface_fd(T, K, params, grid,
                                    dt_target=dt_target,
                                    use_cfl=use_cfl)
        dS = S[1] - S[0]
        dV = V[1] - V[0]
        i0 = int(round(params.S0 / dS))
        j0 = int(round(params.V0 / dV))
        prices[i] = float(F0[i0, j0])

    return param_grid, prices
