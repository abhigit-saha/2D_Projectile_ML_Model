# CARS Project — Projectile Trajectory & Velocity Prediction
## ML + Computer Vision 

---

## Project Summary
This system implements the full RSQR (HEMRL/RSQR/CARS/EDS/2024) pipeline:
- Computer vision detection of ANY projectile from video
- Multiple ML models: PINN, LSTM/GRU, Kalman Filter, Parabolic Regression
- Velocity estimation and trajectory prediction
- Quantitative comparison (RMSE, MAE, ADE, FDE) — ML vs physics baselines
- Works for sports, defence (mortar/artillery), robotics

---

## Supported Objects (9 types)

| Key            | Object                  | Mass   | Presets                         |
|----------------|-------------------------|--------|---------------------------------|
| `table_tennis` | Table Tennis Ball        | 2.7 g  | topspin, push, counter          |
| `tennis`       | Tennis Ball              | 57.7 g | serve, topspin_drive, slice     |
| `football`     | Football (Soccer)        | 430 g  | power_shot, banana, pass        |
| `basketball`   | Basketball               | 620 g  | jump_shot, free_throw, full_court |
| `cricket`      | Cricket Ball             | 156 g  | fast, spin, medium              |
| `baseball`     | Baseball                 | 145 g  | fastball, curveball, changeup   |
| `volleyball`   | Volleyball               | 270 g  | spike, float_serve, rally       |
| `mortar`       | Mortar / Artillery Shell | 4.2 kg | high_velocity, medium, low      |
| `generic`      | Custom / Any Object      | 100 g  | high_angle, low_fast, lob       |

---

## Install

```bash
pip install numpy scipy matplotlib opencv-python
```
No PyTorch needed — all models run in pure NumPy/SciPy.

---

## Usage

### List all objects
```bash
python main.py --list
```

### Simulate any object (no video needed)
```bash
python main.py --simulate --object tennis --preset serve
python main.py --simulate --object football --preset banana
python main.py --simulate --object mortar --preset high_velocity
python main.py --simulate --object cricket --preset spin
```

### Analyse real video
```bash
python main.py --video myvideo.mp4 --object tennis
python main.py --video game.mp4   --object football --color white
python main.py --video clip.mp4   --object basketball --annotate-video
```

### Run all objects (full comparison demo)
```bash
python main.py --demo --output demo_results
```

---

## All Options

| Flag              | Default       | Description                                   |
|-------------------|---------------|-----------------------------------------------|
| `--object`        | table_tennis  | Object type to analyse                        |
| `--preset`        | first preset  | Simulation scenario                           |
| `--video`         | None          | Input video path                              |
| `--color`         | auto          | Detection colour: white/orange/yellow/green   |
| `--output`        | results       | Output folder                                 |
| `--simulate`      | off           | Use synthetic simulation                      |
| `--demo`          | off           | Run all objects + comparison                  |
| `--list`          | off           | Print all objects                             |
| `--annotate-video`| off           | Write annotated output MP4                    |
| `--pinn-iters`    | 5000          | PINN max iterations                           |
| `--lstm-epochs`   | 500           | LSTM training iterations                      |
| `--noise`         | 0.005         | Noise std (m) for simulation                  |
| `--no-bg-sub`     | off           | Disable background subtraction                |
| `--max-frames`    | all           | Limit frames processed                        |

---

## Output Files (saved in results/)

| File               | Contents                                              |
|--------------------|-------------------------------------------------------|
| `full_report.png`  | Combined: 2D traj, x(t), y(t), loss curves, kinematics, speed |
| `comparison.png`   | All models on same plot                               |
| `metrics.png`      | RMSE / ADE / FDE bar chart per model                  |
| `velocity.png`     | Speed profile + PINN estimated V₀                     |
| `annotated.mp4`    | Input video with PINN + Kalman overlays               |

---

## File Structure

```
cars_project/
├── main.py         ← Run this
├── config.py       ← All object definitions (add new ones here)
├── physics.py      ← ODE simulator + Kalman filter + parabolic baseline
├── models.py       ← PINN + LSTM/GRU + metrics
├── detector.py     ← OpenCV-based object tracker
├── visualiser.py   ← All plots + annotated video writer
└── README.md
```

---

## Models Implemented

### 1. PINN (Physics-Informed Neural Network)
- Maps time t → (x(t), y(t)) via MLP
- Loss = data loss + physics residual (Newton's 2nd law)
- Learns CD and ω simultaneously with network weights
- Based on: Chiha et al. CBMI 2024 + HEMRL RSQR

### 2. LSTM / GRU Predictor
- Sliding window of past positions → future k positions
- Implemented as GRU cell in pure NumPy (no PyTorch)
- Learns temporal patterns in trajectory data

### 3. Kalman Filter
- Constant-acceleration model: state = [x, y, vx, vy, ax, ay]
- Online position + velocity estimation from noisy measurements
- Predicts ahead using state transition matrix

### 4. Parabolic Regression (Classical Baseline)
- Fits y = ax² + bx + c to observed positions
- Physics baseline for comparison with ML models

---

## Physics Model (2D)

```
ax = -k·V·CD·vx  +  Magnus_x(ω)
ay = -g  -  k·V·CD·vy  +  Magnus_y(ω)

k  = ρ·A / (2·m)          (scales automatically per object)
Magnus = CL·k·V × ω       (only when has_spin=True)
CL = 1 / (2 + (R·ω/V)⁻¹) (Stepanek 1988)
```

---

## Adding a New Object (e.g. Golf Ball)

Add one entry to `OBJECTS` in `config.py`:

```python
"golf": {
    "display_name": "Golf Ball",
    "mass_kg":      0.0459,
    "radius_m":     0.0214,
    "drag_coeff":   0.24,
    "has_spin":     True,
    "colors":       ["white"],
    "min_radius_px": 3, "max_radius_px": 20,
    "scene_width_m": 200.0,
    "classify":     lambda V0, rps: "Drive" if V0>40 else "Short Game",
    "presets": {
        "drive": dict(V0=70.0, angle_deg=12.0, omega=50.0, t_end=6.0),
    },
},
```

No other file needs to change.

---

## Metrics

| Metric | Full Name              | What it measures              |
|--------|------------------------|-------------------------------|
| RMSE_x | Root Mean Sq Error x   | Horizontal prediction accuracy|
| RMSE_y | Root Mean Sq Error y   | Vertical prediction accuracy  |
| ADE    | Average Displacement Error | Mean position error       |
| FDE    | Final Displacement Error   | Error at end of trajectory|

---

## Milestones (RSQR Alignment)

| Quarter | Deliverable                    | This code          |
|---------|--------------------------------|--------------------|
| Q1      | Literature + problem formulation | README + config.py |
| Q2      | Data simulation + preprocessing  | physics.py + detector.py |
| Q3      | Model development + training     | models.py          |
| Q4      | Testing + evaluation + report    | main.py + visualiser.py |
