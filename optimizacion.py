# optimizacion.py

import numpy as np
from scipy.optimize import minimize

from diferencias_finitas import ModelParams, GridParams, price_surface_fd
from monte_carlo import mc_price_scott

def target_prices_MCHigh(TK_list, params: ModelParams,
                         N_hi: int = 800, W_hi: int = 100_000,
                         seed: int = 2025):
    """
    Genera precios objetivo (ground truth) con Monte Carlo de alta fidelidad.
    """
    out = []
    for (T, K) in TK_list:
        prices_R, stats = mc_price_scott(
            T, K,
            params=params,
            N=N_hi,
            W=W_hi,
            R=1,
            seed=seed,
            clip_V=True
        )
        out.append(stats["mean"])
    return np.array(out, dtype=float)

def fd_prices_at_S0V0(TK_list, params: ModelParams,
                      grid: GridParams,
                      dt_target: float | None,
                      use_cfl: bool = True):
    """
    Calcula F_FD(0,S0,V0) en todos los (T,K) de TK_list.
    """
    prices = []
    for (T, K) in TK_list:
        F0, S, V = price_surface_fd(T, K, params, grid,
                                    dt_target=dt_target,
                                    use_cfl=use_cfl)
        dS = S[1] - S[0]
        dV = V[1] - V[0]
        i0 = int(round(params.S0 / dS))
        j0 = int(round(params.V0 / dV))
        prices.append(float(F0[i0, j0]))
    return np.array(prices, dtype=float)

def make_objective_1param(TK_list,
                          P_target: np.ndarray,
                          params_base: ModelParams,
                          grid: GridParams,
                          param_name: str,
                          bounds: tuple[float, float],
                          dt_target: float | None,
                          use_cfl: bool = True,
                          lam: float = 0.0,
                          prior: float | None = None):
    """
    Construye J(x) para optimizar UN solo parámetro (param_name).
    J(x) = || F_FD(x) - P_target ||^2 + lam*(x-prior)^2
    """
    def J(x_arr: np.ndarray):
        x = float(x_arr[0])
        x = float(np.clip(x, bounds[0], bounds[1]))

        # clon de los parámetros
        params = ModelParams(**vars(params_base))
        setattr(params, param_name, x)

        # precios FD con ese parámetro
        P_fd = fd_prices_at_S0V0(TK_list, params, grid,
                                 dt_target=dt_target,
                                 use_cfl=use_cfl)

        resid = P_fd - P_target
        loss = float(np.dot(resid, resid))

        if prior is not None and lam > 0.0:
            loss += lam * (x - prior)**2

        return loss
    return J

def calibrate_one_param(TK_list,
                        params_base: ModelParams,
                        grid: GridParams,
                        param_name: str,
                        bounds: tuple[float, float],
                        dt_target: float | None,
                        use_cfl: bool = True,
                        N_hi: int = 800,
                        W_hi: int = 100_000,
                        seed: int = 2025):
    """
    Calibra un parámetro (param_name) minimizando J frente a precios MC-HF.
    """
    P_target = target_prices_MCHigh(TK_list, params_base,
                                    N_hi=N_hi, W_hi=W_hi, seed=seed)
    J = make_objective_1param(TK_list, P_target,
                              params_base, grid,
                              param_name=param_name,
                              bounds=bounds,
                              dt_target=dt_target,
                              use_cfl=use_cfl)

    x0 = np.array([getattr(params_base, param_name)], dtype=float)

    res = minimize(J, x0,
                   method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6, "disp": True})

    x_opt = float(np.clip(res.x[0], bounds[0], bounds[1]))
    params_opt = ModelParams(**vars(params_base))
    setattr(params_opt, param_name, x_opt)

    return params_opt, res, P_target
