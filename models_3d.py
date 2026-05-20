import numpy as np
import torch
import torch.nn as nn
from physics import G, RHO

class _PINNNet3D(nn.Module):
    """Inner PyTorch network: t (scalar, normalised) → [X_n, Y_n, Z_n]"""
    def __init__(self, hidden, neurons):
        super().__init__()
        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers += [nn.Linear(neurons, 3)]  # Output X, Y, Z
        self.net = nn.Sequential(*layers)

        # Learnable physics parameters
        self.log_CD  = nn.Parameter(torch.tensor(0.0))   # log(CD)
        self.log_om  = nn.Parameter(torch.tensor(3.4))   # log(omega) ≈ 30 rad/s

        # Xavier init
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


class ProjectilePINN3D:
    """
    3-D PINN: maps t → (X(t), Y(t), Z(t)) in Camera Coordinate Frame.
    Z is forward, X is right, Y is DOWN.
    Gravity acts in the +Y direction.
    """

    def __init__(self, cfg, K_matrix, hidden=4, neurons=64,
                 beta=1e-3, n_coll=300, max_iter=5000, seed=42):
        self.cfg      = cfg
        self.K        = torch.tensor(K_matrix, dtype=torch.float32)
        self.hidden   = hidden
        self.neurons  = neurons
        self.beta     = beta
        self.n_coll   = n_coll
        self.max_iter = max_iter
        self.seed     = seed
        self._trained = False
        self.loss_history = []

    def _norm_t(self, t):
        return (t - self._t0) / (self._dt + 1e-12)

    def fit(self, t_obs, u_obs, v_obs, verbose=True, direction=None):
        """
        t_obs: timestamps (s)
        u_obs: pixel X coordinates
        v_obs: pixel Y coordinates
        """
        torch.manual_seed(self.seed)
        cfg = self.cfg

        # --- Time Normalisation ---
        self._t0 = float(t_obs.min())
        self._dt = float(t_obs.max() - t_obs.min())
        tn = torch.tensor(self._norm_t(t_obs), dtype=torch.float32).reshape(-1, 1)

        # --- Observations (Pixels) ---
        u_t = torch.tensor(u_obs, dtype=torch.float32)
        v_t = torch.tensor(v_obs, dtype=torch.float32)
        
        # We don't normalise spatial outputs of the network heavily here 
        # because the projection loss handles the pixel scale. 
        # But we'll scale the network outputs to represent meters roughly.
        # Let's say network outputs are naturally around [-1, 1], we scale by a factor
        # to make optimization easier (e.g., 20 meters max flight).
        self.scale_X = 20.0
        self.scale_Y = 20.0
        self.scale_Z = 50.0  # Z is depth, might be larger

        # --- Collocation points ---
        tc = torch.linspace(0.0, 2.0, self.n_coll, requires_grad=True).reshape(-1, 1)

        # --- Physics scaling ---
        K_drag = RHO * np.pi * cfg["radius_m"]**2 / (2 * cfg["mass_kg"])
        R_ball = cfg["radius_m"]

        # --- Build model + optimiser ---
        net = _PINNNet3D(self.hidden, self.neurons)
        if not cfg["has_spin"]:
            net.log_om.requires_grad_(False)

        # Standard robust optimizer with learning rate decay
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_iter, eta_min=1e-5)

        mse = nn.MSELoss()
        hist = []

        if verbose:
            print(f"\n[PINN 3D] Fitting {cfg['display_name']} | {len(t_obs)} pts")

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        for epoch in range(self.max_iter):
            optimizer.zero_grad()

            # ── 1. Data loss (Projection) ────────────────────────────────────
            out_obs = net(tn)
            X_pred = out_obs[:, 0] * self.scale_X
            Y_pred = out_obs[:, 1] * self.scale_Y
            Z_pred = out_obs[:, 2] * self.scale_Z + 1.0  # Add 1.0 to prevent Z=0 division

            # Pinhole camera projection
            u_pred = (fx * X_pred / Z_pred) + cx
            v_pred = (fy * Y_pred / Z_pred) + cy

            loss_data = mse(u_pred, u_t) + mse(v_pred, v_t)

            # ── 2. Physics loss (exact autograd derivatives) ─────────────────
            tc_fresh = tc.detach().clone().requires_grad_(True)
            out_c = net(tc_fresh)
            
            Xc = out_c[:, 0:1] * self.scale_X
            Yc = out_c[:, 1:2] * self.scale_Y
            Zc = out_c[:, 2:3] * self.scale_Z

            # First derivatives (velocities)
            ts = self._dt
            dX_dt = torch.autograd.grad(Xc, tc_fresh, grad_outputs=torch.ones_like(Xc), create_graph=True)[0] / ts
            dY_dt = torch.autograd.grad(Yc, tc_fresh, grad_outputs=torch.ones_like(Yc), create_graph=True)[0] / ts
            dZ_dt = torch.autograd.grad(Zc, tc_fresh, grad_outputs=torch.ones_like(Zc), create_graph=True)[0] / ts

            # Second derivatives (accelerations)
            d2X_dt2 = torch.autograd.grad(dX_dt, tc_fresh, grad_outputs=torch.ones_like(dX_dt), create_graph=True)[0] / ts
            d2Y_dt2 = torch.autograd.grad(dY_dt, tc_fresh, grad_outputs=torch.ones_like(dY_dt), create_graph=True)[0] / ts
            d2Z_dt2 = torch.autograd.grad(dZ_dt, tc_fresh, grad_outputs=torch.ones_like(dZ_dt), create_graph=True)[0] / ts

            V = torch.sqrt(dX_dt**2 + dY_dt**2 + dZ_dt**2) + 1e-9
            CD = torch.exp(net.log_CD)

            # Projectile ODE residual
            # Note: in OpenCV camera coords, Y is down. Gravity is positive on Y.
            res_X = d2X_dt2 + K_drag * V * CD * dX_dt
            res_Y = d2Y_dt2 - G + K_drag * V * CD * dY_dt   # Gravity pulls DOWN (+Y in OpenCV)
            res_Z = d2Z_dt2 + K_drag * V * CD * dZ_dt

            # Spin (Magnus) - Simplified 3D spin (assuming spin mostly around X-axis for top-spin)
            if cfg["has_spin"]:
                om = torch.exp(net.log_om)
                CL = 1.0 / (2.0 + (R_ball * om / V)**(-1) + 1e-9)
                # If topspin around X axis, lift is along Y axis (actually -Y, pointing UP)
                res_Y = res_Y + K_drag * V * (CL / om) * (om * dZ_dt)  # Lift pushes UP (-Y)
                res_Z = res_Z - K_drag * V * (CL / om) * (om * dY_dt)

            loss_phys = (res_X**2).mean() + (res_Y**2).mean() + (res_Z**2).mean()

            # ── 3. Directional Prior Penalty ─────────────────────────────────
            loss_dir = 0.0
            if direction == "away":
                # Penalty if dZ_dt < 0 (moving towards)
                loss_dir = torch.mean(torch.relu(-dZ_dt)**2) * 10000.0
            elif direction == "towards":
                # Penalty if dZ_dt > 0 (moving away)
                loss_dir = torch.mean(torch.relu(dZ_dt)**2) * 10000.0

            # ── 4. Total loss ────────────────────────────────────────────────
            # Because pixel MSE is very large (e.g. 100^2), we might need to scale beta
            loss = loss_data + (self.beta * 1000) * loss_phys + loss_dir

            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 500 == 0:
                hist.append(float(loss.item()))
                if verbose:
                    print(f"  [PINN 3D] epoch {epoch:4d} | Total: {loss.item():.1f} | Data: {loss_data.item():.1f} | Phys: {loss_phys.item():.3f} | CD={CD.item():.3f}")

        self._net = net
        self._trained = True
        return self

    def predict_3d(self, t_query):
        self._net.eval()
        tn = torch.tensor(self._norm_t(np.asarray(t_query, dtype=np.float64)), dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            out = self._net(tn).numpy()
            
        X = out[:, 0] * self.scale_X
        Y = out[:, 1] * self.scale_Y
        Z = out[:, 2] * self.scale_Z + 1.0
        return X, Y, Z

class AlgebraicProjectile3D:
    """
    Direct algebraic optimizer for 3D Projectile Motion.
    Fits exact equations of motion to 2D pixels, guaranteed to be a parabola.
    """
    def __init__(self, K_matrix):
        self.K = np.array(K_matrix)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.G = 9.81
        self.theta_opt = None

    def fit(self, t_obs, u_obs, v_obs, direction=None):
        from scipy.optimize import least_squares
        
        t = np.asarray(t_obs, dtype=np.float64)
        u = np.asarray(u_obs, dtype=np.float64)
        v = np.asarray(v_obs, dtype=np.float64)
        
        # 1. Analytical Initialization
        # Fit parabola to Y-pixels: v(t) = C*t^2 + B*t + A
        poly_v = np.polyfit(t, v, 2)
        C, B, A = poly_v
        
        # C = (fy * 0.5 * G) / Z0  => Z0 = (fy * 0.5 * G) / C
        Z0_guess = (self.fy * 0.5 * self.G) / max(C, 1e-3)
        # Cap unreasonable depth guesses
        Z0_guess = np.clip(Z0_guess, 0.5, 50.0)
        
        # Fit line to X-pixels: u(t) = M*t + K
        poly_u = np.polyfit(t, u, 1)
        M, K = poly_u
        
        Vx_guess = M * Z0_guess / self.fx
        X0_guess = (K - self.cx) * Z0_guess / self.fx
        
        Vy_guess = B * Z0_guess / self.fy
        Y0_guess = (A - self.cy) * Z0_guess / self.fy
        
        Vz_guess = 0.0
        if direction == "away": Vz_guess = 5.0
        if direction == "towards": Vz_guess = -5.0
        
        theta_guess = [X0_guess, Y0_guess, Z0_guess, Vx_guess, Vy_guess, Vz_guess]
        print(f"  [Algebraic 3D] Analytical Z0 guess: {Z0_guess:.2f} m")

        # 2. Optimization setup
        def residual(theta):
            X0, Y0, Z0, Vx, Vy, Vz = theta
            X = X0 + Vx * t
            Y = Y0 + Vy * t + 0.5 * self.G * t**2
            Z = np.maximum(Z0 + Vz * t, 0.1)
            
            u_pred = self.fx * (X / Z) + self.cx
            v_pred = self.fy * (Y / Z) + self.cy
            
            return np.concatenate([u - u_pred, v - v_pred])

        # Bounds: Ensure Z0 > 0. Apply directional prior to Vz if requested.
        lower_bounds = [-np.inf, -np.inf, 0.1, -np.inf, -np.inf, -np.inf]
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        
        if direction == "away":
            lower_bounds[5] = 0.0  # Vz must be positive
        elif direction == "towards":
            upper_bounds[5] = 0.0  # Vz must be negative

        # 3. Solve
        res = least_squares(residual, theta_guess, bounds=(lower_bounds, upper_bounds))
        self.theta_opt = res.x
        
        print(f"  [Algebraic 3D] Fit complete. RMSE pixels: {np.mean(res.fun**2)**0.5:.2f}")
        return self

    def predict_3d(self, t_query):
        X0, Y0, Z0, Vx, Vy, Vz = self.theta_opt
        t = np.asarray(t_query, dtype=np.float64)
        
        X = X0 + Vx * t
        Y = Y0 + Vy * t + 0.5 * self.G * t**2
        Z = np.maximum(Z0 + Vz * t, 0.1)
        
        return X, Y, Z

class LinearProjectile3D:
    """
    Linear Algebraic Optimizer based on:
    "Online 3-D Trajectory Estimation of a Flying Object from a Monocular Image Sequence"
    (R. Herrejon et al., IROS 2009).
    """
    def __init__(self, K_matrix, **kwargs):
        self.K = np.array(K_matrix)
        self.fu = self.K[0, 0]
        self.fv = self.K[1, 1]
        self.cu = self.K[0, 2]
        self.cv = self.K[1, 2]
        self.g = 9.81
        self.C = None

    def fit(self, t_obs, u_obs, v_obs, direction=None, **kwargs):
        N = len(t_obs)
        H = np.zeros((2 * N, 8))
        q = np.zeros(2 * N)
        
        for i in range(N):
            t = t_obs[i]
            # Center the pixel coordinates according to equations (7), (8), (11), (12)
            ui = u_obs[i] - self.cu
            vi = v_obs[i] - self.cv
            
            # We normalize by d * C6 = 1 (Y-acceleration) because gravity acts primarily in the Y-axis for upright cameras.
            # Unknowns: a = [a1, a2, a3, a4, a5, a7, a8, a9]
            
            # Equation 1 (u_i): fu*a1 + fu*t*a2 + fu*t^2*a3 - ui*a7 - ui*t*a8 - ui*t^2*a9 = 0
            H[2*i]   = [self.fu, self.fu*t, self.fu*t**2, 0, 0, -ui, -ui*t, -ui*t**2]
            q[2*i]   = 0.0
            
            # Equation 2 (v_i): fv*a4 + fv*t*a5 - vi*a7 - vi*t*a8 - vi*t^2*a9 = -fv*t^2
            H[2*i+1] = [0, 0, 0, self.fv, self.fv*t, -vi, -vi*t, -vi*t**2]
            q[2*i+1] = -self.fv * t**2

        # Solve linear system Ha = q using Batch Least Squares
        a, residuals, rank, s = np.linalg.lstsq(H, q, rcond=None)
        
        a1, a2, a3, a4, a5, a7, a8, a9 = a
        
        # Calculate scale factor d from gravity constraint (22)
        # C3 = a3/d, C6 = 1/d, C9 = a9/d
        d = 2 * np.sqrt(a3**2 + 1.0 + a9**2) / self.g
        
        # Z0 = C7 = a7 / d. The object must be in front of the camera (Z > 0)
        # If it's negative, we flip the sign of d.
        if (a7 / d) < 0:
            d = -d
            
        # Recover physical parameters C_1 to C_9
        self.C = np.zeros(9)
        self.C[0:5] = a[0:5] / d
        self.C[5] = 1.0 / d
        self.C[6:9] = a[5:8] / d
        
        # Print findings
        print(f"  [Linear 3D] Method: Paper 1619 (Herrejon et al.)")
        print(f"  [Linear 3D] --- INTERMEDIATE DEBUG ---")
        print(f"  [Linear 3D] Scale factor (d): {d:.5f}")
        print(f"  [Linear 3D] Raw a-vector (Linear Weights):")
        print(f"    a1(X):{a1:.2f} a2(Vx):{a2:.2f} a3(Ax/2):{a3:.2f}")
        print(f"    a4(Y):{a4:.2f} a5(Vy):{a5:.2f} (a6=1 for norm)")
        print(f"    a7(Z):{a7:.2f} a8(Vz):{a8:.2f} a9(Az/2):{a9:.2f}")
        
        print(f"  [Linear 3D] Recovered Physical C-vector:")
        print(f"    C1(X0): {self.C[0]:.2f}m | C2(Vx0): {self.C[1]:.2f}m/s | C3(Ax/2): {self.C[2]:.2f}m/s^2")
        print(f"    C4(Y0): {self.C[3]:.2f}m | C5(Vy0): {self.C[4]:.2f}m/s | C6(Ay/2): {self.C[5]:.2f}m/s^2")
        print(f"    C7(Z0): {self.C[6]:.2f}m | C8(Vz0): {self.C[7]:.2f}m/s | C9(Az/2): {self.C[8]:.2f}m/s^2")
        
        # Calculate min/max Y over the observed time
        y_vals = self.C[3] + self.C[4] * t_obs + self.C[5] * t_obs**2
        print(f"  [Linear 3D] Calculated Y (Height) range: from {np.min(y_vals):.2f}m to {np.max(y_vals):.2f}m")
        print(f"  [Linear 3D] Z0 (Depth): {self.C[6]:.2f}m")
        print(f"  [Linear 3D] Vz (Towards/Away): {self.C[7]:.2f}m/s")
        print(f"  [Linear 3D] --------------------------")
        
        # Note: direction hint is mathematically not needed for this method, 
        # as Z > 0 perfectly constrains the ambiguity!
        
        return self

    def predict_3d(self, t_query):
        t = np.asarray(t_query, dtype=np.float64)
        
        # Equations (4), (5), (6)
        X = self.C[0] + self.C[1] * t + self.C[2] * t**2
        Y = self.C[3] + self.C[4] * t + self.C[5] * t**2
        Z = self.C[6] + self.C[7] * t + self.C[8] * t**2
        
        return X, Y, Z
