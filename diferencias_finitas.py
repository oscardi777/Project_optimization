# diferencias_finitas.py

import numpy as np
from dataclasses import dataclass

@dataclass
class ModelParams:
    r: float
    kappa_star: float
    theta_star: float
    eta_star: float
    delta: float
    rho: float
    S0: float
    V0: float

@dataclass
class GridParams:
    Smax: float = 200.0
    Vmax: float = 0.16
    dS: float   = 1.0
    dV: float   = 0.0013

def build_grid(grid: GridParams):
    NS = int(round(grid.Smax / grid.dS))
    NV = int(round(grid.Vmax / grid.dV))
    S  = np.linspace(0.0, grid.Smax, NS+1, dtype=np.float64)
    V  = np.linspace(0.0, grid.Vmax, NV+1, dtype=np.float64)
    return S, V

def stable_dt(S, V, params: ModelParams, safety: float = 0.25):
    """
    Estimador de Δt estable (tipo CFL) para esquema explícito.
    No viene del paper, pero es muy útil para evitar explosiones.
    """
    dS = S[1] - S[0]
    dV = V[1] - V[0]
    S_mat, V_mat = np.meshgrid(S, V, indexing="ij")

    aSS = 0.5 * V_mat * (S_mat**2)
    aVV = 0.5 * (params.delta**2) * V_mat
    aSV = np.abs(params.rho) * params.delta * V_mat * S_mat
    bS  = params.r * S_mat
    bV  = np.abs(params.kappa_star * (params.theta_star - (np.sqrt(V_mat) - params.eta_star)**2))
    c0  = np.abs(params.r)

    lam = (aSS / (dS**2)
           + aVV / (dV**2)
           + aSV / (2.0 * dS * dV)
           + bS  / (2.0 * dS)
           + bV  / (2.0 * dV)
           + c0)

    Lmax = float(np.nanmax(lam))
    return safety / (Lmax + 1e-14)

def apply_boundaries(F_curr, F_next, S, dS, dV, dt, params: ModelParams):
    """
    Aplica las condiciones de frontera del paper:
    (14) Neumann en S=Smax
    (15) F(t,0,V)=0
    (16) cond. reducida en V=0
    (17) F(t,S,Vmax)=S
    """
    NS, NV = F_curr.shape[0] - 1, F_curr.shape[1] - 1

    # (15) S = 0
    F_curr[0, :] = 0.0

    # (14) S = Smax  -> dF/dS = 1
    F_curr[NS, :] = F_curr[NS - 1, :] + dS

    # (17) V = Vmax -> F = S
    F_curr[:, NV] = S

    # (16) V = 0 -> condición reducida
    if NV >= 1 and NS >= 2:
        FS0 = (F_next[2:, 0] - F_next[:-2, 0]) / (2.0 * dS)         # centrada en S
        FV0 = (F_next[1:-1, 1] - F_next[1:-1, 0]) / dV              # adelante en V
        F_curr[1:-1, 0] = F_next[1:-1, 0] + dt * (
            - params.r * F_next[1:-1, 0]
            + params.r * S[1:-1] * FS0
            + params.kappa_star * (params.theta_star - params.eta_star**2) * FV0
        )

def step_explicit(F_next, S, V, dt, params: ModelParams):
    """
    UN paso de tiempo hacia atrás con el esquema explícito de Scott.
    Usa las derivadas centradas y el término mixto F_SV.
    """
    NS, NV = F_next.shape[0] - 1, F_next.shape[1] - 1
    dS = S[1] - S[0]
    dV = V[1] - V[0]

    # Coeficientes de la PDE (13), escritos para L = -RHS
    S_mat, V_mat = np.meshgrid(S, V, indexing="ij")
    aSS = 0.5 * V_mat * (S_mat**2)
    aVV = 0.5 * (params.delta**2) * V_mat
    aSV = params.rho * params.delta * V_mat * S_mat
    bS  = params.r * S_mat
    bV  = params.kappa_star * (params.theta_star - (np.sqrt(V_mat) - params.eta_star)**2)
    c0  = -params.r

    F_curr = F_next.copy()
    apply_boundaries(F_curr, F_next, S, dS, dV, dt, params)

    # slices del interior
    C   = F_next[1:-1, 1:-1]
    Cp1 = F_next[2:  , 1:-1]
    Cm1 = F_next[:-2 , 1:-1]
    Cjp = F_next[1:-1, 2:  ]
    Cjm = F_next[1:-1, :-2 ]
    Cpp = F_next[2:  , 2:  ]
    Cpm = F_next[2:  , :-2 ]
    Cmp = F_next[:-2 , 2:  ]
    Cmm = F_next[:-2 , :-2 ]

    # derivadas centradas
    FS  = (Cp1 - Cm1) / (2.0 * dS)
    FV  = (Cjp - Cjm) / (2.0 * dV)
    FSS = (Cp1 - 2.0 * C + Cm1) / (dS**2)
    FVV = (Cjp - 2.0 * C + Cjm) / (dV**2)
    FSV = (Cpp - Cpm - Cmp + Cmm) / (4.0 * dS * dV)

    # coeficientes en el interior
    aSS_i = aSS[1:-1, 1:-1]
    aVV_i = aVV[1:-1, 1:-1]
    aSV_i = aSV[1:-1, 1:-1]
    bS_i  = bS [1:-1, 1:-1]
    bV_i  = bV [1:-1, 1:-1]
    c0_i  = c0

    # L = -RHS
    L = ( c0_i * C
          + bS_i * FS + bV_i * FV
          + aSS_i * FSS + aVV_i * FVV + aSV_i * FSV )

    F_curr[1:-1, 1:-1] = C + dt * L
    apply_boundaries(F_curr, F_next, S, dS, dV, dt, params)
    np.maximum(F_curr, 0.0, out=F_curr)
    return F_curr

def price_surface_fd(T: float, K: float,
                     params: ModelParams,
                     grid: GridParams,
                     dt_target: float | None = None,
                     use_cfl: bool = True):
    """
    Resuelve la PDE de Scott vía diferencias finitas explícitas.
    Devuelve:
      F0: matriz F(0,S,V)
      S, V: rejillas espaciales
    """
    S, V = build_grid(grid)
    dS = S[1] - S[0]
    dV = V[1] - V[0]

    # Paso temporal
    if dt_target is None:
        dt = stable_dt(S, V, params)
    else:
        dt = dt_target
        if use_cfl:
            dt_cfl = stable_dt(S, V, params)
            dt = min(dt, dt_cfl)

    Nt = max(1, int(np.ceil(T / dt)))
    dt = T / Nt

    NS, NV = len(S) - 1, len(V) - 1
    F_next = np.zeros((NS+1, NV+1), dtype=np.float64)

    # Payoff terminal: F(T,S,V) = max(S-K,0)
    payoff = np.maximum(S - K, 0.0)
    F_next[:] = payoff[:, None]

    for _ in range(Nt):
        F_next = step_explicit(F_next, S, V, dt, params)

    return F_next, S, V
