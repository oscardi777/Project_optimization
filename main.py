# main.py

from diferencias_finitas import ModelParams, GridParams, price_surface_fd
from monte_carlo import mc_price_scott
from optimizacion import calibrate_one_param, fd_prices_at_S0V0, target_prices_MCHigh
from sensibilidad import sensibilidad_objetivo_1param, sensibilidad_precio_1param

import numpy as np
import pandas as pd
import time

def run_diferencias_finitas():
    params = ModelParams(
        r=0.05,
        kappa_star=1.98,
        theta_star=0.05,
        eta_star=0.0101,
        delta=0.02,
        rho=0.07,
        S0=100.0,
        V0=0.04
    )
    grid = GridParams(Smax=200.0, Vmax=0.16, dS=1.0, dV=0.0013)

    T = 1.0
    K = 100.0
    dt_target = 1.5024e-4  # valor de la tabla del paper

    F0, S, V = price_surface_fd(T, K, params, grid,
                                dt_target=dt_target,
                                use_cfl=True)
    dS = S[1] - S[0]
    dV = V[1] - V[0]
    i0 = int(round(params.S0 / dS))
    j0 = int(round(params.V0 / dV))
    print(f"FD: F(0,S0={params.S0},V0={params.V0}) ≈ {F0[i0,j0]:.6f}")

def run_monte_carlo():
    params = ModelParams(
        r=0.05,
        kappa_star=1.98,
        theta_star=0.05,
        eta_star=0.0101,
        delta=0.02,
        rho=0.07,
        S0=100.0,
        V0=0.04
    )
    T = 1.0
    K = 100.0
    prices_R, stats = mc_price_scott(
        T, K,
        params=params,
        N=200,
        W=10_000,
        R=50,
        seed=12345
    )
    print(f"MC: mean={stats['mean']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}")

def run_optimizacion_y_sensibilidad():
    # 0. Tiempo ejecucion:
    start_time = time.time()  # ⏱️ inicio

    # 1. Parámetros base
    params_base = ModelParams(
        r=0.05,
        kappa_star=1.98,
        theta_star=0.05,
        eta_star=0.0101,
        delta=0.02,
        rho=0.07,
        S0=100.0,
        V0=0.04
    )
    grid = GridParams(Smax=200.0, Vmax=0.16, dS=1.0, dV=0.0013)
    dt_target = 1.5024e-4

    # puntos de calibración
    TK_list = [(0.25,60), (0.25,80), (0.25,100),
               (0.5,80), (1.0,100), (1.0,120)]

    bounds = (0.01, 5.0)

    # 2. Calibrar kappa_star contra precios objetivo MC-HF
    params_opt, res, P_target = calibrate_one_param(
        TK_list,
        params_base,
        grid,
        param_name="kappa_star",
        bounds=bounds,
        dt_target=dt_target,
        use_cfl=True,
        N_hi=800,
        W_hi=50_000,
        seed=2025
    )

    kappa_opt = params_opt.kappa_star
    print("\n*** Resultado de la calibración ***")
    print(f"kappa_star óptimo = {kappa_opt:.6f}")
    print(f"J(kappa_star óptimo) = {res.fun:.6e}")
    print("Precios target (MC-HF):", P_target)

    # 3. Sensibilidades conjuntas: J(kappa) y precio(kappa) en la MISMA malla

    # Rango relativo alrededor de kappa*
    factor_min, factor_max = 0.7, 1.3   # ±30 %
    n_points = 25

    # 3.1 Sensibilidad del funcional J(kappa)
    param_grid_J, J_values = sensibilidad_objetivo_1param(
        TK_list,
        P_target,
        params_opt,
        grid,
        param_name="kappa_star",
        bounds=bounds,
        dt_target=dt_target,
        use_cfl=True,
        factor_min=factor_min,
        factor_max=factor_max,
        n_points=n_points
    )

    # 3.2 Sensibilidad del precio en un (T,K) fijo
    T_sens, K_sens = 1.0, 100.0
    param_grid_P, prices = sensibilidad_precio_1param(
        T_sens,
        K_sens,
        params_opt,
        grid,
        param_name="kappa_star",
        bounds=bounds,
        dt_target=dt_target,
        use_cfl=True,
        factor_min=factor_min,
        factor_max=factor_max,
        n_points=n_points
    )

    param_grid_J = np.array(param_grid_J, dtype=float)
    param_grid_P = np.array(param_grid_P, dtype=float)
    J_values = np.array(J_values, dtype=float)
    prices = np.array(prices, dtype=float)

    # Comprobar que las dos mallas de kappa coinciden
    if not np.allclose(param_grid_J, param_grid_P):
        print("⚠️ Aviso: las mallas de kappa para J y precio no coinciden exactamente.")
    kappa_grid = param_grid_J  # son iguales en la práctica

    # Índice del kappa óptimo dentro de la malla
    idx_opt = int(np.argmin(np.abs(kappa_grid - kappa_opt)))
    price_opt = prices[idx_opt]
    J_opt_grid = J_values[idx_opt]

    # (1) Imprimir el valor objetivo y su precio
    print("\n*** Valor óptimo en la malla de sensibilidad ***")
    print(f"kappa_star* = {kappa_opt:.6f}")
    print(f"J(kappa_star*) en la malla = {J_opt_grid:.6e}")
    print(f"Precio FD en T={T_sens}, K={K_sens}: {price_opt:.6f}")

    # (2) Tabla única con TODAS las kappas, J(kappa), precio(kappa) y error |precio - precio_opt|
    df = pd.DataFrame({
        "kappa": kappa_grid,
        "delta_kappa_%": 100.0 * (kappa_grid / kappa_opt - 1.0),
        "J": J_values,
        "precio": prices
    })

    # Error absoluto del precio respecto al precio en el kappa óptimo
    df["error_abs_precio"] = np.abs(df["precio"] - price_opt)

    # Marcar explícitamente cuál es el punto óptimo
    df["es_optimo"] = False
    df.loc[idx_opt, "es_optimo"] = True

    # Ordenar por variación relativa de kappa
    df = df.sort_values("delta_kappa_%").reset_index(drop=True)

    pd.set_option("display.float_format", lambda x: f"{x: .6f}")

    print("\n*** Sensibilidad conjunta de J y del precio respecto a kappa_star (±30%) ***")
    print(df.to_string(index=False))

    # 0. Tiempo ejecucion:
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n⏱️ Tiempo total de ejecución: {elapsed:.2f} segundos")


if __name__ == "__main__":
    # Algunas funciones disponibles para correr el codigo:
    # run_diferencias_finitas() # Ejecuta el método de diferencias finitas
    # run_monte_carlo() # Ejecuta el método de Monte Carlo
    run_optimizacion_y_sensibilidad() # Ejecuta la optimización y análisis de sensibilidad
    pass
