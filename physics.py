"""
physics.py
----------
Ground-truth ODE simulator + classical baselines.

Provides:
  simulate()          - Runge-Kutta 2-D trajectory (drag + Magnus)
  parabolic_fit()     - classical parabolic regression baseline
  kalman_tracker()    - Kalman filter for online position/velocity estimation
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize  import curve_fit

G   = 9.81   # m/s²
RHO = 1.2    # kg/m³


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _k(mass, radius):
    return RHO * np.pi * radius**2 / (2 * mass)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ODE Simulator (Runge-Kutta)
# ─────────────────────────────────────────────────────────────────────────────

def simulate(cfg, V0, angle_deg, omega=0.0, t_end=1.0, n_pts=120,
             noise_m=0.0, seed=0):
    """
    Simulate 2-D projectile trajectory.

    Parameters
    ----------
    cfg       : object config dict
    V0        : launch speed (m/s)
    angle_deg : launch angle above horizontal
    omega     : spin rate (rad/s)
    t_end     : flight duration (s)
    n_pts     : output samples
    noise_m   : Gaussian noise std (metres) added to positions
    seed      : random seed for noise

    Returns
    -------
    t, x, y, vx, vy : np.ndarray  (all in SI units)
    """
    K  = _k(cfg["mass_kg"], cfg["radius_m"])
    CD = cfg["drag_coeff"]
    R  = cfg["radius_m"]
    has_spin = cfg["has_spin"]

    ang = np.radians(angle_deg)
    vx0 = V0 * np.cos(ang)
    vy0 = V0 * np.sin(ang)

    def odes(t, s):
        x, y, vx, vy = s
        V = np.sqrt(vx**2 + vy**2) + 1e-9

        ax = -K * V * CD * vx
        ay = -G - K * V * CD * vy

        if has_spin and abs(omega) > 1e-6:
            CL = 1.0 / (2.0 + (R * abs(omega) / V)**(-1) + 1e-9)
            on = abs(omega) + 1e-9
            ax +=  K * V * (CL / on) * ( omega * vy)
            ay +=  K * V * (CL / on) * (-omega * vx)

        return [vx, vy, ax, ay]

    # Stop when ball hits ground (y < 0) after leaving it
    def hit_ground(t, s): return s[1]
    hit_ground.terminal  = True
    hit_ground.direction = -1

    t_eval = np.linspace(0, t_end, n_pts)
    sol = solve_ivp(odes, (0, t_end), [0.0, 0.0, vx0, vy0],
                    t_eval=t_eval, events=hit_ground,
                    method="RK45", max_step=1e-3, dense_output=False)

    t  = sol.t
    x  = sol.y[0]
    y  = sol.y[1]
    vx = sol.y[2]
    vy = sol.y[3]

    if noise_m > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0, noise_m, x.shape)
        y = y + rng.normal(0, noise_m, y.shape)

    return t, x, y, vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# 2. Parabolic Regression Baseline  (classical physics-only)
# ─────────────────────────────────────────────────────────────────────────────

def parabolic_fit(x_obs, y_obs, x_future):
    """
    Fit y = ax² + bx + c to observed (x,y) and predict at x_future.
    Returns y_pred, coefficients (a,b,c).
    """
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c

    try:
        popt, _ = curve_fit(parabola, x_obs, y_obs, maxfev=10000)
        return parabola(x_future, *popt), popt
    except Exception:
        return np.full_like(x_future, np.nan), (np.nan, np.nan, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kalman Filter  (constant-acceleration model)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanTracker:
    """
    2-D Kalman filter for projectile tracking.
    State: [x, y, vx, vy, ax, ay]
    Measurement: [x, y]

    Usage
    -----
    kf = KalmanTracker(dt=1/fps)
    for (mx, my) in measurements:
        x, y, vx, vy = kf.update(mx, my)
    x_pred, y_pred = kf.predict_ahead(k_steps)
    """

    def __init__(self, dt=1/30, process_noise=1.0, measure_noise=5.0):
        self.dt = dt
        n = 6   # state dim
        m = 2   # measurement dim

        # State transition matrix (constant acceleration)
        dt2 = 0.5 * dt**2
        self.F = np.array([
            [1, 0, dt, 0,  dt2, 0  ],
            [0, 1, 0,  dt, 0,   dt2],
            [0, 0, 1,  0,  dt,  0  ],
            [0, 0, 0,  1,  0,   dt ],
            [0, 0, 0,  0,  1,   0  ],
            [0, 0, 0,  0,  0,   1  ],
        ])

        # Measurement matrix
        self.H = np.zeros((m, n))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Noise covariances
        self.Q = np.eye(n) * process_noise
        self.R = np.eye(m) * measure_noise**2

        # Initial state and covariance
        self.x = np.zeros(n)
        self.P = np.eye(n) * 500.0
        self._initialised = False

    def update(self, mx, my):
        """Feed one measurement, return smoothed (x, y, vx, vy)."""
        z = np.array([mx, my])

        if not self._initialised:
            self.x[0] = mx
            self.x[1] = my
            self._initialised = True
            return mx, my, 0.0, 0.0

        # Predict
        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.Q

        # Update
        y_inn = z - self.H @ xp
        S     = self.H @ Pp @ self.H.T + self.R
        K     = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K @ y_inn
        self.P = (np.eye(len(self.x)) - K @ self.H) @ Pp

        return self.x[0], self.x[1], self.x[2], self.x[3]

    def predict_ahead(self, k_steps):
        """Predict k future positions from current state."""
        xs, ys = [], []
        xc = self.x.copy()
        for _ in range(k_steps):
            xc = self.F @ xc
            xs.append(xc[0])
            ys.append(xc[1])
        return np.array(xs), np.array(ys)

    def get_velocity(self):
        return self.x[2], self.x[3]

    def get_speed(self):
        return float(np.sqrt(self.x[2]**2 + self.x[3]**2))
