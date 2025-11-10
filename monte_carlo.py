# monte_carlo.py

import numpy as np
from dataclasses import dataclass

# Si quieres reusar ModelParams, puedes importarlo:
from diferencias_finitas import ModelParams

def mc_price_scott(
    T: float,
    K: float,
    params: ModelParams,
    N: int = 200,          # pasos en el tiempo
    W: int = 10_000,       # trayectorias por corrida
    R: int = 1,            # repeticiones para min/mean/max
    seed: int = 12345,
    clip_V: bool = True,
    batch_size: int | None = None
):
    """
    Monte Carlo para el modelo de Scott bajo Q:
      dS = r S dt + sqrt(V) S dW1
      dV = κ* [θ* - (sqrt(V) - η*)^2] dt + δ sqrt(V) dW2
      corr(dW1, dW2) = ρ

    Devuelve:
      prices_R: array de precios (R repeticiones)
      stats: dict con min, mean, max, std
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    sqrt_dt = np.sqrt(dt)

    if batch_size is None:
        batch_size = W

    prices_rep = np.empty(R, dtype=np.float64)

    for rep in range(R):
        remaining = W
        sum_payoff = 0.0

        while remaining > 0:
            B = min(batch_size, remaining)
            remaining -= B

            S = np.full(B, params.S0, dtype=np.float64)
            V = np.full(B, params.V0, dtype=np.float64)

            for _ in range(N):
                Z1 = rng.standard_normal(B)
                Zp = rng.standard_normal(B)
                Z2 = params.rho * Z1 + np.sqrt(max(1.0 - params.rho**2, 0.0)) * Zp

                if clip_V:
                    V_nonneg = np.maximum(V, 0.0)
                else:
                    V_nonneg = V

                S = S + params.r * S * dt + np.sqrt(V_nonneg) * S * sqrt_dt * Z1
                drift_V = params.kappa_star * (params.theta_star - (np.sqrt(V_nonneg) - params.eta_star)**2)
                V = V + drift_V * dt + params.delta * np.sqrt(V_nonneg) * sqrt_dt * Z2

                if clip_V:
                    V = np.maximum(V, 0.0)

            payoff = np.maximum(S - K, 0.0)
            sum_payoff += payoff.sum()

        prices_rep[rep] = np.exp(-params.r * T) * (sum_payoff / W)

    stats = {
        "min":  float(prices_rep.min()),
        "mean": float(prices_rep.mean()),
        "max":  float(prices_rep.max()),
        "std":  float(prices_rep.std(ddof=1)) if R > 1 else 0.0,
    }
    return prices_rep, stats
