"""
models.py
---------
All ML models for projectile trajectory prediction.

  1. ProjectilePINN  - Physics-Informed Neural Network (NumPy/SciPy)
  2. LSTMPredictor   - Sequence-to-sequence LSTM (NumPy/SciPy, no PyTorch)
  3. predict_all()   - Run all models + baselines, return comparison dict
"""

import numpy as np
from scipy.optimize import minimize
from physics import parabolic_fit, KalmanTracker, G, RHO


# ═════════════════════════════════════════════════════════════════════════════
# Shared MLP utilities (plain NumPy)
# ═════════════════════════════════════════════════════════════════════════════

def _pack(layers):
    return np.concatenate([p.ravel() for wb in layers for p in wb])

def _unpack(theta, sizes):
    layers, i = [], 0
    for a, b in zip(sizes, sizes[1:]):
        W = theta[i:i+a*b].reshape(b, a); i += a*b
        bs= theta[i:i+b];                 i += b
        layers.append((W, bs))
    return layers

def _fwd(t, layers):
    a = t
    for j, (W, b) in enumerate(layers):
        z = a @ W.T + b
        a = np.tanh(z) if j < len(layers)-1 else z
    return a

def _d(t, layers, order=1, h=1e-5):
    if order == 1:
        return (_fwd(t+h, layers) - _fwd(t-h, layers)) / (2*h)
    return (_fwd(t+h, layers) - 2*_fwd(t, layers) + _fwd(t-h, layers)) / h**2

def _xavier(sizes, rng):
    out = []
    for a, b in zip(sizes, sizes[1:]):
        W = rng.normal(0, np.sqrt(2/(a+b)), (b, a))
        out.append((W, np.zeros(b)))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 1. PINN  (Physics-Informed Neural Network)
# ═════════════════════════════════════════════════════════════════════════════

class ProjectilePINN:
    """
    2-D PINN: maps t → (x(t), y(t)).
    Simultaneously learns NN weights + drag coefficient + spin rate.

    Based on: Chiha et al., CBMI 2024
    Extended for any projectile via config dict.
    """

    def __init__(self, cfg, hidden=3, neurons=20,
                 beta=1e-3, n_coll=200, max_iter=5000, seed=42):
        self.cfg      = cfg
        self.hidden   = hidden
        self.neurons  = neurons
        self.beta     = beta
        self.n_coll   = n_coll
        self.max_iter = max_iter
        self.seed     = seed
        self._trained = False
        self.loss_history = []

    def _norm_t(self, t): return (t-self._t0)/(self._dt+1e-12)

    def fit(self, t_obs, x_obs, y_obs, verbose=True):
        rng = np.random.default_rng(self.seed)
        cfg = self.cfg

        self._t0 = t_obs.min(); self._dt = t_obs.max()-t_obs.min()
        self._x0 = x_obs.min(); self._xs = max(x_obs.max()-x_obs.min(),1e-3)
        self._y0 = y_obs.min(); self._ys = max(y_obs.max()-y_obs.min(),1e-3)

        tn   = self._norm_t(t_obs)
        xn   = (x_obs-self._x0)/self._xs
        yn   = (y_obs-self._y0)/self._ys
        tc   = np.linspace(0,1,self.n_coll)
        data = np.stack([xn,yn],1)

        sizes = [1]+[self.neurons]*self.hidden+[2]
        self._sizes = sizes

        K = RHO*np.pi*cfg["radius_m"]**2/(2*cfg["mass_kg"])
        R = cfg["radius_m"]
        CD0 = cfg["drag_coeff"]
        ts, xs_sc, ys_sc = self._dt, self._xs, self._ys

        init = _xavier(sizes, rng)
        lam0 = [np.log(CD0)] + ([np.log(30.)] if cfg["has_spin"] else [])
        theta0 = np.concatenate([_pack(init), lam0])

        n_nn = sum((a+1)*b for a,b in zip(sizes,sizes[1:]))

        def loss(theta):
            layers = _unpack(theta[:n_nn], sizes)
            lam    = theta[n_nn:]
            CD     = np.exp(lam[0])

            # Data loss
            tdc = tn.reshape(-1,1)
            pred= _fwd(tdc, layers)
            Ls  = np.mean((pred-data)**2)

            # Physics loss
            tcc = tc.reshape(-1,1)
            vn  = _d(tcc, layers, 1)
            an  = _d(tcc, layers, 2)
            vx  = vn[:,0:1]*(xs_sc/ts)
            vy  = vn[:,1:2]*(ys_sc/ts)
            ax  = an[:,0:1]*(xs_sc/ts**2)
            ay  = an[:,1:2]*(ys_sc/ts**2)
            V   = np.sqrt(vx**2+vy**2)+1e-9

            fx = ax + K*V*CD*vx
            fy = ay + G + K*V*CD*vy

            if cfg["has_spin"] and len(lam)>1:
                om  = np.exp(lam[1])
                CL  = 1./(2.+(R*om/V)**(-1)+1e-9)
                on  = abs(om)+1e-9
                fx += -K*V*(CL/on)*(om*vy)
                fy +=  K*V*(CL/on)*(om*vx)

            Lf = np.mean((fx/G)**2)+np.mean((fy/G)**2)
            return (1-self.beta)*Ls + self.beta*Lf

        hist=[]; itr=[0]
        def cb(xk):
            itr[0]+=1
            if itr[0]%200==0:
                v=loss(xk); hist.append(v)
                if verbose:
                    lam=xk[n_nn:]
                    print(f"  [PINN] iter {itr[0]:4d}  loss={v:.3e}  "
                          f"CD={np.exp(lam[0]):.3f}")

        if verbose:
            print(f"\n[PINN] Fitting {cfg['display_name']} | "
                  f"{len(t_obs)} pts | spin={cfg['has_spin']}")

        res = minimize(loss, theta0, method="L-BFGS-B", callback=cb,
                       options={"maxiter":self.max_iter,"ftol":1e-15,"gtol":1e-10})

        self._layers = _unpack(res.x[:n_nn], sizes)
        self._lam    = res.x[n_nn:]
        self._n_nn   = n_nn
        self.loss_history = hist
        self._trained = True

        if verbose:
            kin=self.kinematics()
            print(f"[PINN] Done  loss={res.fun:.3e}  "
                  f"V0={kin['speed']:.2f}m/s  CD={kin['CD']:.3f}")
        return self

    def predict(self, t_query):
        tn  = self._norm_t(np.asarray(t_query)).reshape(-1,1)
        out = _fwd(tn, self._layers)
        return out[:,0]*self._xs+self._x0, out[:,1]*self._ys+self._y0

    def kinematics(self):
        CD = float(np.exp(self._lam[0]))
        om = float(np.exp(self._lam[1])) if len(self._lam)>1 else 0.
        t0 = np.array([[0.]])
        vn = _d(t0, self._layers, 1)
        vx = float(vn[0,0])*self._xs/self._dt
        vy = float(vn[0,1])*self._ys/self._dt
        return {"CD": CD, "spin_rps": om/(2*np.pi),
                "vx0": vx, "vy0": vy,
                "speed": float(np.sqrt(vx**2+vy**2))}

    def classify(self):
        kin = self.kinematics()
        tm  = np.array([[0.5]])
        acc = _d(tm, self._layers, 2)
        sign = -np.sign(float(acc[0,1]))
        rps  = sign*kin["spin_rps"]
        return self.cfg["classify"](kin["speed"], rps), rps, kin["speed"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. LSTM Predictor  (sequence → sequence, NumPy/SciPy only)
#    Implemented as a vanilla RNN in NumPy since PyTorch is unavailable.
#    Uses sliding window of past positions to predict next k positions.
# ═════════════════════════════════════════════════════════════════════════════

class LSTMPredictor:
    """
    Simplified LSTM-like sequence predictor implemented in pure NumPy.
    Uses a sliding window of past (x,y) observations to predict future positions.

    Architecture: GRU cell (simpler than LSTM, same idea, fewer params)
                  → Linear output layer
    """

    def __init__(self, window=10, hidden_size=32, predict_steps=20,
                 lr=0.01, epochs=500, seed=42):
        self.window  = window
        self.H       = hidden_size
        self.k       = predict_steps
        self.lr      = lr
        self.epochs  = epochs
        self.seed    = seed
        self._trained = False
        self.loss_history = []

    # ── Sigmoid / tanh ────────────────────────────────────────────────────────
    @staticmethod
    def _sig(x): return 1./(1.+np.exp(-np.clip(x,-30,30)))
    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x,-30,30))

    # ── GRU cell forward ─────────────────────────────────────────────────────
    def _gru(self, x, h, params):
        Wz,Uz,bz, Wr,Ur,br, Wn,Un,bn = params
        z = self._sig(x@Wz.T + h@Uz.T + bz)
        r = self._sig(x@Wr.T + h@Ur.T + br)
        n = self._tanh(x@Wn.T + (r*h)@Un.T + bn)
        return (1-z)*h + z*n

    # ── Initialise parameters ─────────────────────────────────────────────────
    def _init_params(self, rng, input_size):
        H, I = self.H, input_size
        std  = 0.1
        def W(r,c): return rng.normal(0,std,(r,c))
        def b(n):   return np.zeros(n)
        gru_params = [
            W(H,I), W(H,H), b(H),   # z gate
            W(H,I), W(H,H), b(H),   # r gate
            W(H,I), W(H,H), b(H),   # n gate
        ]
        # Output layers: one per future step (shared weights → linear decode)
        Wo = W(2*self.k, H)
        bo = b(2*self.k)
        return gru_params, Wo, bo

    # ── Forward pass ─────────────────────────────────────────────────────────
    def _forward(self, seq, params, Wo, bo):
        """seq: (window, 2)  → returns (k, 2)"""
        h = np.zeros(self.H)
        for t in range(seq.shape[0]):
            h = self._gru(seq[t], h, params)
        out = h @ Wo.T + bo           # (2k,)
        return out.reshape(self.k, 2)

    # ── Build windows ─────────────────────────────────────────────────────────
    def _make_windows(self, x_n, y_n):
        data = np.stack([x_n, y_n], 1)   # (N,2)
        X, Y = [], []
        for i in range(len(data)-self.window-self.k+1):
            X.append(data[i:i+self.window])
            Y.append(data[i+self.window:i+self.window+self.k])
        return np.array(X), np.array(Y)

    # ── Flatten / unflatten params ────────────────────────────────────────────
    def _flat(self, gru, Wo, bo):
        return np.concatenate([p.ravel() for p in gru] +
                               [Wo.ravel(), bo])

    def _unflat(self, v, I):
        H = self.H
        shapes = [(H,I),(H,H),(H,), (H,I),(H,H),(H,), (H,I),(H,H),(H,)]
        gru, idx = [], 0
        for s in shapes:
            n = int(np.prod(s))
            gru.append(v[idx:idx+n].reshape(s)); idx+=n
        Wo = v[idx:idx+2*self.k*H].reshape(2*self.k, H); idx+=2*self.k*H
        bo = v[idx:idx+2*self.k]
        return gru, Wo, bo

    # ── Train ─────────────────────────────────────────────────────────────────
    def fit(self, t_obs, x_obs, y_obs, verbose=True):
        rng = np.random.default_rng(self.seed)

        # Normalise
        self._x0=x_obs.min(); self._xs=max(x_obs.max()-x_obs.min(),1e-3)
        self._y0=y_obs.min(); self._ys=max(y_obs.max()-y_obs.min(),1e-3)
        xn = (x_obs-self._x0)/self._xs
        yn = (y_obs-self._y0)/self._ys

        X, Y = self._make_windows(xn, yn)   # (M, window, 2), (M, k, 2)
        if len(X) < 2:
            if verbose: print("[LSTM] Not enough data for sequence training")
            self._trained = False
            return self

        I = 2  # input size
        gru0, Wo0, bo0 = self._init_params(rng, I)
        theta0 = self._flat(gru0, Wo0, bo0)
        hist   = []

        def loss(theta):
            gru, Wo, bo = self._unflat(theta, I)
            total = 0.
            for xi, yi in zip(X, Y):
                pred = self._forward(xi, gru, Wo, bo)
                total += np.mean((pred-yi)**2)
            return total/len(X)

        itr=[0]
        def cb(xk):
            itr[0]+=1
            if itr[0]%100==0:
                v=loss(xk); hist.append(v)
                if verbose:
                    print(f"  [LSTM] iter {itr[0]:4d}  loss={v:.4e}")

        if verbose:
            print(f"\n[LSTM] Training on {len(X)} windows  "
                  f"(window={self.window}, predict={self.k})")

        res = minimize(loss, theta0, method="L-BFGS-B", callback=cb,
                       options={"maxiter":self.epochs,"ftol":1e-12,"gtol":1e-8})

        self._gru, self._Wo, self._bo = self._unflat(res.x, I)
        self._trained = True
        self.loss_history = hist
        if verbose:
            print(f"[LSTM] Done  loss={res.fun:.4e}")
        return self

    def predict_next(self, x_recent, y_recent):
        """
        Given recent observations, predict next self.k positions.
        x_recent, y_recent : last `window` observations (metres)
        Returns x_fut, y_fut (metres)
        """
        if not self._trained:
            return np.array([]), np.array([])
        w = self.window
        xn = (x_recent-self._x0)/self._xs
        yn = (y_recent-self._y0)/self._ys
        # Pad or trim to window size
        seq = np.stack([xn[-w:], yn[-w:]], 1)
        if len(seq) < w:
            pad = np.zeros((w-len(seq), 2))
            seq = np.vstack([pad, seq])
        pred = self._forward(seq, self._gru, self._Wo, self._bo)
        xf = pred[:,0]*self._xs + self._x0
        yf = pred[:,1]*self._ys + self._y0
        return xf, yf


# ═════════════════════════════════════════════════════════════════════════════
# 3. Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true_x, y_true_y, y_pred_x, y_pred_y):
    """Compute RMSE, MAE, ADE, FDE for trajectory comparison."""
    n    = min(len(y_true_x), len(y_pred_x))
    tx   = y_true_x[:n]; ty = y_true_y[:n]
    px   = y_pred_x[:n]; py = y_pred_y[:n]
    dist = np.sqrt((tx-px)**2 + (ty-py)**2)
    rmse_x = float(np.sqrt(np.mean((tx-px)**2)))
    rmse_y = float(np.sqrt(np.mean((ty-py)**2)))
    mae_x  = float(np.mean(np.abs(tx-px)))
    mae_y  = float(np.mean(np.abs(ty-py)))
    ade    = float(np.mean(dist))
    fde    = float(dist[-1]) if n > 0 else np.nan
    return {"RMSE_x":rmse_x,"RMSE_y":rmse_y,
            "MAE_x":mae_x,"MAE_y":mae_y,
            "ADE":ade,"FDE":fde}


# ═════════════════════════════════════════════════════════════════════════════
# 4. Unified prediction runner
# ═════════════════════════════════════════════════════════════════════════════

def predict_all(cfg, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
                pinn_iters=5000, lstm_epochs=500, verbose=True):
    """
    Train all models and return a comparison dict.

    Returns
    -------
    dict with keys: 'pinn', 'lstm', 'kalman', 'parabolic'
    Each value: dict with 'x_pred', 'y_pred', 'metrics', 'label', 'kin'
    """
    results = {}
    t_query = t_gt   # predict on ground-truth time axis

    # ── 1. PINN ───────────────────────────────────────────────────────────────
    pinn = ProjectilePINN(cfg, max_iter=pinn_iters)
    pinn.fit(t_obs, x_obs, y_obs, verbose=verbose)
    xp, yp = pinn.predict(t_query)
    label, rps, V0 = pinn.classify()
    results["pinn"] = {
        "model":   pinn,
        "x_pred":  xp, "y_pred": yp,
        "metrics": compute_metrics(x_gt, y_gt, xp, yp),
        "label":   label,
        "kin":     pinn.kinematics(),
        "loss":    pinn.loss_history,
    }

    # ── 2. LSTM ───────────────────────────────────────────────────────────────
    lstm = LSTMPredictor(window=min(10,len(t_obs)//3),
                         predict_steps=len(t_query),
                         epochs=lstm_epochs)
    lstm.fit(t_obs, x_obs, y_obs, verbose=verbose)
    if lstm._trained:
        xl, yl = lstm.predict_next(x_obs, y_obs)
        # LSTM predicts absolute positions (future), align to query length
        n = min(len(xl), len(t_query))
        xl_full = np.full(len(t_query), np.nan)
        yl_full = np.full(len(t_query), np.nan)
        xl_full[-n:] = xl[:n]
        yl_full[-n:] = yl[:n]
        results["lstm"] = {
            "x_pred": xl_full, "y_pred": yl_full,
            "metrics": compute_metrics(x_gt[-n:], y_gt[-n:], xl[:n], yl[:n]),
            "loss":    lstm.loss_history,
        }
    else:
        results["lstm"] = {"x_pred": np.array([]), "y_pred": np.array([]),
                           "metrics": {}, "loss": []}

    # ── 3. Kalman Filter ──────────────────────────────────────────────────────
    dt = float(np.mean(np.diff(t_obs))) if len(t_obs)>1 else 1/30
    kf = KalmanTracker(dt=dt)
    kx, ky = [], []
    for xi, yi in zip(x_obs, y_obs):
        sx, sy, _, _ = kf.update(xi, yi)
        kx.append(sx); ky.append(sy)
    # Predict ahead to fill t_query
    n_ahead = max(0, len(t_query)-len(kx))
    if n_ahead > 0:
        xah, yah = kf.predict_ahead(n_ahead)
        kx_full = np.concatenate([kx, xah])
        ky_full = np.concatenate([ky, yah])
    else:
        kx_full = np.array(kx[:len(t_query)])
        ky_full = np.array(ky[:len(t_query)])
    results["kalman"] = {
        "x_pred":  kx_full, "y_pred": ky_full,
        "metrics": compute_metrics(x_gt, y_gt, kx_full, ky_full),
    }

    # ── 4. Parabolic Regression (classical baseline) ──────────────────────────
    x_fut = np.interp(t_query, t_obs,
                      np.linspace(x_obs.min(), x_obs.max(), len(t_obs)))
    y_par, coeffs = parabolic_fit(x_obs, y_obs, x_fut)
    results["parabolic"] = {
        "x_pred":  x_fut, "y_pred": y_par,
        "metrics": compute_metrics(x_gt, y_gt, x_fut, y_par),
        "coeffs":  coeffs,
    }

    return results
