#set document(title: "CARS: Computer-Assisted Reconstitution of Shots")
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10.5pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set par(justify: true, leading: 0.65em)
#show heading.where(level: 1): it => {
  v(0.6em)
  it
  v(0.2em)
}

// ── Placeholder image helper ──────────────────────────────────────────────────
#let placeholder(w: 100%, h: 5cm, label: "[Placeholder Image]") = {
  rect(
    width: w, height: h,
    fill: luma(230), stroke: 1pt + luma(160),
    radius: 4pt,
    align(center + horizon)[
      #text(size: 11pt, fill: luma(100), style: "italic")[#label]
    ]
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Cover Page
// ─────────────────────────────────────────────────────────────────────────────
#page(numbering: none)[
  #align(center)[
    #v(1.2cm)
    #image("media/mnnit.jpg", width: 5.5cm)
    #v(0.5cm)
    #text(size: 13pt, weight: "bold")[Motilal Nehru National Institute of Technology Allahabad]
    #v(0.1cm)
    #text(size: 10pt)[Prayagraj, Uttar Pradesh: 211004]
    #v(0.8cm)
    #line(length: 85%, stroke: 0.7pt)
    #v(0.6cm)
    #text(size: 14pt, weight: "bold")[
      Development of a Computer Vision Framework for\ Motion Feature Extraction of a Projectile\ using Physics-Informed Neural Networks
    ]
    #v(0.4cm)
    #text(size: 10pt, style: "italic")[Group Project Technical Report]
    #v(0.6cm)
    #line(length: 85%, stroke: 0.7pt)
    #v(0.8cm)

    #text(size: 9.5pt, fill: luma(60))[*Submitted by*]
    #v(0.4cm)
    #table(
      columns: (1fr, 1fr, 1fr),
      align: center,
      stroke: none,
      inset: 6pt,
      [*Abhijit Saha*\ #text(size: 9pt, fill: luma(60))[20236004]],
      [*Shivam Raj Srivastav*\ #text(size: 9pt, fill: luma(60))[20230051]],
      [*Vidhi Singh*\ #text(size: 9pt, fill: luma(60))[20238027]],
    )
    #v(0.3cm)
    #text(size: 9.5pt, fill: luma(60))[Group 7]

    #v(1.2cm)
    #text(size: 9.5pt, fill: luma(60))[*Under the Guidance of*]
    #v(0.3cm)
    #text(size: 11pt, weight: "bold")[Dr. Abhishek Kumar Tiwari]
    #v(0.1cm)
  ]
]

// ─────────────────────────────────────────────────────────────────────────────
// Table of Contents
// ─────────────────────────────────────────────────────────────────────────────
#outline(depth: 3, indent: 1.5em)
#pagebreak()


// =============================================================================
// 1. INTRODUCTION
// =============================================================================
= Introduction

Reconstructing the flight path of a projectile from a standard video recording is a deceptively challenging problem. The camera provides only 2D pixel coordinates, the image is contaminated by background clutter and player motion, and the underlying physics involves non-linear aerodynamic forces that have no convenient closed-form solution.

The *CARS* (Computer-Assisted Reconstitution of Shots) pipeline addresses this challenge through a tightly integrated four-stage approach:

+ *Computer Vision*: Automatic per-frame detection of the ball using background subtraction and shape-based filtering, producing a raw time-stamped pixel trajectory.
+ *Data Cleaning*: Removal of false detections (player limbs, shadows) via an interactive lasso tool or automated RANSAC-based filtering.
+ *Physics-Informed Neural Network (PINN)*: A neural network that learns the trajectory while simultaneously satisfying the aerodynamic differential equations governing drag and Magnus spin lift, enabling identification of physical quantities like drag coefficient $C_D$ and spin rate $omega$.
+ *Comparative Evaluation*: The PINN result is benchmarked against two classical methods (Kalman Filter and Parabolic Regression) using standard trajectory error metrics.

The pipeline was validated across *12 independent experiments* covering multiple rally shots and serve conditions, all recorded at 1920×1080 @ 30 fps. Object types supported include: Tennis Ball, Table Tennis, Football, Basketball, Cricket Ball, Baseball, Volleyball, Mortar Shell, and a Generic object.

== Motivation

While the system is demonstrated on sports footage, the underlying problem of reconstructing a ballistic trajectory from monocular video has direct relevance to *defence applications*. A few concrete use cases drove the design choices:

- *Threat assessment*: Knowing the speed and drag-corrected trajectory of an incoming projectile in real time allows an immediate estimate of its danger level and likely impact zone, without relying on radar.

- *Launch origin reconstruction*: Post-event, the fitted trajectory can be extrapolated backward in time to narrow down the likely launch point, which is useful for counter-battery analysis.

- *Uncertainty-aware impact prediction*: Because the PINN is a continuous function, confidence intervals on its output can be propagated to produce a probabilistic impact zone rather than a single predicted point.

- *Command support*: Extracted kinematic parameters (speed, spin, drag coefficient) give analysts a compact, physics-grounded summary of observed projectile behaviour that feeds into broader threat models.

The sports setting provided a controlled, repeatable environment for developing and validating the pipeline before deployment in more operationally sensitive contexts.

// =============================================================================
// 2. BACKGROUND
// =============================================================================
= Background: Physics of Projectile Flight

== Classical (Vacuum) Motion

In an idealised vacuum, a projectile launched with speed $V_0$ at angle $theta$ follows a perfect parabolic arc governed by:

$ x(t) = V_0 cos(theta) dot.c t, quad y(t) = V_0 sin(theta) dot.c t - frac(1,2) g t^2 $

where $g = 9.81 "m/s"^2$. This model is analytically simple but insufficient for real sports balls, where aerodynamic effects are substantial.

== Aerodynamic Drag

Moving through air of density $rho = 1.2 "kg/m"^3$, a sphere of mass $m$, radius $r$, and drag coefficient $C_D$ experiences a retarding force:

$ bold(F)_"drag" = -frac(1,2) rho C_D pi r^2 |bold(v)| bold(v) $

Defining the aerodynamic scale factor $K = rho pi r^2 slash (2m)$, the equations of motion become:

$ dot.double(x) = -K C_D |bold(v)| dot(x), quad dot.double(y) = -g - K C_D |bold(v)| dot(y) $

where $|bold(v)| = sqrt(dot(x)^2 + dot(y)^2)$ is the instantaneous speed.

== Magnus Spin Lift

A spinning ball generates a lateral Magnus force perpendicular to its velocity. The spin lift coefficient is:

$ C_L = frac(1, 2 + (r omega slash |bold(v)|)^(-1)) $

Adding the Magnus terms, the full coupled ODE system is:

$ dot.double(x) = -K C_D |bold(v)| dot(x) + K |bold(v)| frac(C_L, omega) (omega dot(y)) $
$ dot.double(y) = -g - K C_D |bold(v)| dot(y) - K |bold(v)| frac(C_L, omega) (omega dot(x)) $

This system has no closed-form solution for arbitrary $C_D$ and $omega$, which is the core motivation for the PINN approach described in Section 4.

// =============================================================================
// 3. COMPUTER VISION PIPELINE
// =============================================================================
= Computer Vision Detection Pipeline

The detection pipeline converts a raw video into a sequence of time-stamped metric positions $(t_i, x_i, y_i)$.

== Step 1: MOG2 Background Subtraction

The first step separates the moving ball from the static background. The Gaussian Mixture of Gaussians (MOG2) algorithm builds an adaptive statistical model of the background by observing each pixel over a history of 500 frames. Pixels that deviate significantly from their learned background distribution are flagged as *foreground*. This produces a binary foreground mask at each frame.

The key advantage of MOG2 is its ability to adapt to gradual scene changes (e.g., lighting shifts, crowd movement) while still reacting quickly to fast-moving objects like a tennis ball. The foreground mask is used to gate subsequent colour-based detection, dramatically reducing false positives from static scene elements.

#figure(
  image("media/before-and-after-mog2.png", width: 100%),
  caption: [Left: raw video frame. Center: MOG2 foreground mask isolating the ball and player motion. Right: The video after applying morphological filtering. The ball appears as a small bright blob.]
)

The raw foreground mask from MOG2 is rarely clean. It typically contains scattered white pixels from sensor noise, thin streaks from motion blur, and small isolated patches where the background model has not yet settled. To deal with this, two morphological operations are applied in sequence.

First, an *erosion* pass shrinks all foreground regions by sliding a small elliptical kernel (radius 4 pixels) over the mask and keeping only pixels where the kernel fits entirely within a foreground region. This wipes out isolated specks and thin lines that are too small to be the ball.

After erosion, some portions of the actual ball blob may have been trimmed away. A *dilation* pass (with a slightly larger kernel, radius 9 pixels) is then applied to grow the remaining regions back outward. The net effect is that real blobs the ball, the racket, the player arm are restored to roughly their original size, while the noise that was removed by erosion does not come back because there was nothing left to dilate.

== Step 2: HSV Colour Thresholding

The foreground mask is further refined by converting the frame from BGR to HSV colour space and applying per-colour range filters. HSV separates *Hue* (colour identity) from *Saturation* and *Value* (illumination), making the filter robust to shadows and varying court lighting.

The detector supports a configurable set of target colours (white, orange, yellow, green, red, brown, black, blue). When running in `--color any` mode, the colour filter is bypassed and only shape information is used.

== Step 3: Circularity Filtering and Blob Selection

After morphological cleanup (erosion to remove noise, dilation to fill gaps), contours are extracted from the combined mask. For each contour, a *circularity score* is computed based on the ratio of its area to the square of its perimeter: a perfect circle scores 1.0. Only blobs scoring above 0.6 are considered, and a strict area gate (1 to 300 px²) rejects large non-ball objects such as player arms or score boards.

The centroid of the highest-scoring circular blob in each frame is recorded as the ball detection for that frame.

== Step 4: Scale Calibration and Coordinate Conversion

Since camera intrinsics are generally unknown, a *manual scale calibration* is performed once per video: the user clicks two points of known real-world separation (e.g., the 1.75 m length of a tennis racket). The pixel distance between the clicks divided by the known real-world distance gives a scale factor $s$ in px/m.

Pixel coordinates $(x_"px", y_"px")$ are then converted to metric coordinates. The $y$-axis is flipped since pixel row indices increase downward while physical height increases upward:

$ x_m = frac(x_"px" - x_"min", s), quad y_m = frac(H_"frame" - y_"px", s) $

#figure(
  image("media/scale.png", width: 100%),
  caption: [The calibration window shows the first video frame. The user clicks two points at a known real-world distance; the computed scale is printed and used for all subsequent conversions.]
)

// =============================================================================
// 4. DATA FILTERING
// =============================================================================
= Data Filtering

Even after the three-stage detector, the raw trajectory contains a number of false detections: typically player arms, racket edges, or background clutter that briefly passes the circularity filter. Two filtering strategies are implemented.

== Interactive Lasso Tool

An OpenCV window plots all detected positions on a *Time vs Vertical Position* ($t$-$y$) axes (rather than $x$-$y$). This choice is deliberate: player limbs that cross the scene at the wrong time appear as disconnected, non-parabolic clusters when viewed as $y$ vs $t$, making them visually easy to identify and remove.

The user can:
- *Left-click* near a point to toggle it on/off individually.
- *Left-click-and-drag* to draw a free-form polygon (lasso); all enclosed points are immediately removed.

After editing, only the toggled-on points are passed to the model training step.

#figure(
  image("media/interactive.png", width: 80%),
  caption: [The lasso tool displays all detected positions in $t$-$y$ space. Blue points are kept; red points are discarded. The user draws a polygon around noise clusters to remove them in bulk.]
)

== RANSAC-Based Automatic Filtering

For automated runs (without user interaction), a *RANSAC* (Random Sample Consensus) algorithm identifies the dominant downward parabolic arc in the $y(t)$ data and discards everything outside it. The algorithm works as follows:

+ Randomly sample 3 detection points.
+ Fit a quadratic $y = a t^2 + b t + c$ through them.
+ *Reject* the fit if $a >= 0$ (an upward parabola is physically impossible under gravity).
+ Count how many of all $N$ points fall within $tau = 0.5$ m of this parabola: these are the *inliers*.
+ Repeat 1500 times; keep the model with the most inliers.

The physics constraint $a < 0$ is the key innovation: it ensures the algorithm only accepts trajectories consistent with a gravitational arc, making it robust against noise bursts and rising artefacts.

#figure(
  image("media/ransac.png", width: 100%),
  caption: [Left: raw $y$ vs $t$ scatter with heavy noise. Right: RANSAC inliers (blue) with the fitted downward parabola (red). Grey points are rejected noise.]
)

// =============================================================================
// 5. TRAJECTORY PREDICTION MODELS
// =============================================================================
= Trajectory Prediction Models

Three models are trained and compared on every experiment run.

== Physics-Informed Neural Network (PINN)

=== Architecture

The PINN is a 4-hidden-layer fully-connected network with 32 $tanh$ neurons per layer, mapping normalised time $tilde(t) in [0,1]$ to normalised 2D position $[tilde(x), tilde(y)]$. Two additional *learnable scalar parameters*: $log C_D$ and $log omega$: are trained alongside the network weights, representing the aerodynamic drag coefficient and spin rate respectively. Positivity is guaranteed via the exponential: $C_D = e^(log C_D)$.

All inputs and outputs are linearly normalised to the unit interval to improve numerical conditioning.

=== Loss Function

The total loss balances two objectives:

*Data loss*: the network must fit the observed $(t_i, x_i, y_i)$ detections:

$ cal(L)_"data" = frac(1, N) sum_(i=1)^N [(tilde(x)_theta (tilde(t)_i) - tilde(x)_i)^2 + (tilde(y)_theta (tilde(t)_i) - tilde(y)_i)^2] $

*Physics loss*: the network's output must satisfy the aerodynamic ODE residuals at 300 additional *collocation points* spread across the trajectory window. Exact derivatives $dot(x), dot.double(x), dot(y), dot.double(y)$ are computed analytically using PyTorch's `autograd` engine, ensuring no approximation error:

$ cal(L)_"phys" = frac(1, N_c) sum_(j=1)^(N_c) (R_x^2(hat(t)_j) + R_y^2(hat(t)_j)) $

where $R_x$ and $R_y$ are the ODE residuals from Section 2.3. The combined loss is:

$ cal(L) = (1 - beta) cal(L)_"data" + beta cal(L)_"phys", quad beta = 10^(-3) $

The small $beta$ ensures the network first fits the measured data precisely, while the physics residual acts as a soft regulariser preventing physically impossible trajectory shapes.

=== Optimisation

The network is trained with the Adam optimiser (initial learning rate $eta_0 = 5 times 10^(-3)$, 5000 epochs). To prevent oscillation in the later training stages and allow fine convergence, the learning rate is annealed using a *cosine schedule*: the rate smoothly decays following the shape of a cosine curve from $eta_0$ down to $eta_"min" = 10^(-5)$:

$ eta_k = eta_"min" + frac(1, 2)(eta_0 - eta_"min")(1 + cos(frac(pi k, K_"max"))) $

This avoids the abrupt drops of step-decay schedules and empirically yields more stable convergence for physics-constrained problems.

=== Output: Equations of Motion and Physical Parameters

After training, the PINN reports:
- *Initial velocity* $(v_(x 0), v_(y 0))$ extracted via autograd at $t = 0$, giving the linearised equations of motion $x(t) approx x_0 + v_(x 0) t$ and $y(t) approx y_0 + v_(y 0) t - 1/2 g t^2$.
- *Drag coefficient* $C_D = e^(log C_D)$: a physically interpretable measure of aerodynamic resistance.
- *Spin rate* $omega = e^(log omega)$ in rad/s, converted to revolutions per second (rps) for reporting.
- *Launch speed* $V_0 = sqrt(v_(x 0)^2 + v_(y 0)^2)$ in m/s.

== Kalman Filter

The Kalman Filter [5] is a well-established recursive Bayesian estimator that produces minimum-variance state estimates by combining a dynamic motion model with noisy measurements. It is widely used in object tracking because it naturally handles missing detections and measurement noise without storing the full observation history.

In this pipeline the Kalman Filter serves as a *classical tracking baseline*. The 6-dimensional state vector captures position, velocity, and acceleration:

$ bold(s) = [x, y, dot(x), dot(y), dot.double(x), dot.double(y)]^T $

=== Prediction Step

At each frame the filter first *predicts* the next state using a constant-acceleration kinematic model. The state transition matrix $bold(F)$ propagates position, velocity, and acceleration forward by $Delta t$:

$ bold(s)_(k|k-1) = bold(F) bold(s)_(k-1|k-1), quad bold(P)_(k|k-1) = bold(F) bold(P)_(k-1|k-1) bold(F)^T + bold(Q) $

where $bold(P)$ is the state covariance matrix and $bold(Q) = sigma_q^2 bold(I)_6$ ($sigma_q = 1.0$) is the process noise covariance, accounting for model imperfections such as the true aerodynamic deceleration that the constant-acceleration assumption ignores.

=== Update Step

When a detection $bold(z)_k = [x_k, y_k]^T$ is available, the filter *corrects* the prediction using the Kalman gain $bold(K)_k$:

$ bold(K)_k = bold(P)_(k|k-1) bold(H)^T (bold(H) bold(P)_(k|k-1) bold(H)^T + bold(R))^(-1) $
$ bold(s)_(k|k) = bold(s)_(k|k-1) + bold(K)_k (bold(z)_k - bold(H) bold(s)_(k|k-1)) $
$ bold(P)_(k|k) = (bold(I) - bold(K)_k bold(H)) bold(P)_(k|k-1) $

Here $bold(H) in RR^(2 times 6)$ is the measurement matrix that extracts $(x, y)$ from the full state, and $bold(R) = sigma_r^2 bold(I)_2$ ($sigma_r = 5.0$ m) is the measurement noise covariance. The Kalman gain automatically weights the prediction and measurement according to their relative uncertainties: a large $bold(R)$ causes the filter to trust its own motion model more; a large $bold(Q)$ causes it to trust the measurement more.

=== Initialisation and Extrapolation

The filter is initialised with $bold(s)_0 = [x_0, y_0, 0, 0, 0, 0]^T$ and a diagonal covariance $bold(P)_0 = 100 bold(I)_6$ to reflect high initial uncertainty. After processing all observed frames, future positions are extrapolated by repeated application of $bold(F)$ without any measurement update, relying entirely on the estimated velocity and acceleration at the last detected frame.

The key limitation of the Kalman Filter here is that its constant-acceleration model does not encode the direction of gravity or drag. It therefore tends to drift upward in regions of high vertical acceleration change and is outperformed by the PINN once the trajectory curves significantly. See [2] for a comparative analysis of Kalman-based trackers on fast-moving objects.

== Parabolic Regression

The classical vacuum-physics baseline fits a spatial parabola $y = a x^2 + b x + c$ directly to the observed $(x, y)$ coordinate pairs using nonlinear least squares (Levenberg--Marquardt). Horizontal position at query times is obtained by linear interpolation from the first to the last detected $x$ position (preserving the direction of travel). This model makes no attempt to account for drag or spin and serves as the lower-accuracy reference.

== Measured vs PINN Velocity

The velocity analysis plots show two distinct velocity estimates plotted together: *Measured* and *PINN*: which are computed by fundamentally different methods and are worth distinguishing clearly.

*Measured velocity* is obtained by simple *first-order finite differences* applied directly to the cleaned position data. For consecutive detections at times $t_i$ and $t_{i+1}$:

$ v_x^"meas"(t_m) = frac(x_(i+1) - x_i, t_(i+1) - t_i), quad v_y^"meas"(t_m) = frac(y_(i+1) - y_i, t_(i+1) - t_i) $

where $t_m = (t_i + t_{i+1})/2$ is the midpoint time. This is a direct, model-free estimate. Its accuracy depends entirely on how densely and regularly the ball was detected: if frames were dropped or a detection was slightly mislocated, the finite difference amplifies that positional error into a large velocity spike.

*PINN velocity* is computed by applying PyTorch's `autograd` engine to differentiate the trained network output analytically with respect to time:

$ v_x^"PINN"(t) = frac(partial tilde(x), partial tilde(t)) dot.c frac(x_"scale", t_"scale") $

Because the PINN has learned a smooth, physics-consistent trajectory: one that must simultaneously satisfy the aerodynamic ODEs: its velocity is inherently smooth and physically plausible everywhere, including at times between actual detections. The PINN velocity profile therefore represents the *best physically-constrained estimate* of the true velocity, while the measured velocity provides the *raw empirical signal* from the camera.

// =============================================================================
// 6. EVALUATION METRICS
// =============================================================================
= Evaluation Metrics

Each model is evaluated against the observed trajectory using four standard metrics:

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: left,
    stroke: 0.4pt,
    table.header([*Metric*], [*Formula*], [*What it measures*]),
    [#math.equation[$"RMSE"_x$, $"RMSE"_y$]],
      [$sqrt(frac(1,n) sum_i (z_i^"gt" - z_i^"pred")^2)$],
      [Per-axis average squared error],
    [ADE],
      [$frac(1,n) sum_i sqrt(Delta x_i^2 + Delta y_i^2)$],
      [Mean position error over whole trajectory],
    [FDE],
      [$sqrt(Delta x_n^2 + Delta y_n^2)$],
      [Position error at the final point only],
  ),
  caption: [Evaluation metrics used to compare all three models]
)

// =============================================================================
// 7. SYSTEM OVERVIEW
// =============================================================================
= System Overview

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: left,
    stroke: 0.4pt,
    table.header([*Stage*], [*Method*], [*Output*]),
    [Detection],       [MOG2 + HSV + Circularity],    [$(t_i, x_i, y_i)$ in pixels],
    [Calibration],     [Manual two-click scale],       [Scale $s$ in px/m],
    [Coordinate Conv.],[Flip $y$, divide by $s$],      [$(t_i, x_i, y_i)$ in metres],
    [Pre-clean],       [IQR outlier gate],             [Reduced outlier set],
    [Filtering],       [Lasso (interactive) / RANSAC], [Clean trajectory],
    [PINN],            [PyTorch + Adam + autograd],    [$C_D$, $omega$, $V_0$, trajectory],
    [Kalman],          [Constant-accel. KF],           [Smoothed + extrapolated],
    [Parabolic],       [Spatial curve fit $y(x)$],     [Vacuum baseline],
    [Evaluation],      [RMSE, ADE, FDE],               [Comparison table + plots],
  ),
  caption: [End-to-end CARS pipeline stages]
)

#pagebreak()

// =============================================================================
// 8. EXPERIMENTS
// =============================================================================

// =============================================================================
// 9. APPENDIX: EXPERIMENTAL RESULTS
// =============================================================================
= Appendix: Experimental Results

Each subsection below shows the four output plots generated by the CARS pipeline for the corresponding experiment: (1) the full multi-panel report, (2) the trajectory comparison across models, (3) the velocity analysis, and (4) the error metrics bar chart.

// ── Helper ────────────────────────────────────────────────────────────────────
#let projsec(num, id) = {
  let base = "results/" + id
  heading(level: 2, "Experiment " + str(num))
  v(0.3em)

  figure(
    image(base + "/full_report.png", width: 100%),
    caption: "Full pipeline report — Experiment " + str(num)
  )
  figure(
    image(base + "/comparison.png", width: 100%),
    caption: "Trajectory comparison (PINN / Kalman / Parabolic) — Experiment " + str(num)
  )
  figure(
    image(base + "/velocity.png", width: 100%),
    caption: "Velocity analysis — Experiment " + str(num)
  )
  figure(
    image(base + "/metrics.png", width: 100%),
    caption: "Error metrics — Experiment " + str(num)
  )
  v(1em)
}

#projsec(1, "proj4")
#projsec(2, "proj5")
#projsec(3, "proj6")
#projsec(4, "proj7")
#projsec(5, "proj8")
#projsec(6, "proj9")
#projsec(7, "proj10")
#projsec(8, "proj11")
#projsec(9, "proj12")
#projsec(10, "proj13")
#projsec(11, "proj14")
#projsec(12, "proj15")

// =============================================================================
// REFERENCES
// =============================================================================
#pagebreak()
= References

#set par(hanging-indent: 1.5em, first-line-indent: 0em)

#let ref(label, body) = [#body #label]

*[1]* Chiha, Z., Péteri, R., & Mascarilla, L. (2024). _Predicting 3D Projectile Motion in Table Tennis Using Computer Vision and Physics-Informed Neural Network_. CBMI 2024 — 21st International Conference on Content-Based Multimedia Indexing, Reykjavik, Iceland.

#v(0.4em)
*[2]* Singh, P.R., Gottumukkala, R., & Maida, A. (n.d.). _An Analysis of Kalman Filter Based Object Tracking Methods for Fast-Moving Tiny Objects_. McNeese State University / University of Louisiana at Lafayette.

#v(0.4em)
*[3]* Zhou, Y., et al. (2024). _Vision-Based 3D Reconstruction Methods Using Computer Vision_ — a survey of current approaches to monocular and stereo-based scene reconstruction.

#v(0.4em)
*[4]* Nielsen, N. _Monocular and Stereo Vision Pipelines for Object Detection and Visual Odometry_ [Video tutorials]. YouTube. #link("https://www.youtube.com/@NicolaiAI")

#v(0.4em)
*[5]* Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems. _Journal of Basic Engineering_, 82(1), 35–45. American Society of Mechanical Engineers (ASME).

#v(0.4em)
*[6]* Fischler, M.A., & Bolles, R.C. (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. _Communications of the ACM_, 24(6), 381–395.

#v(0.4em)
*[7]* Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. _Journal of Computational Physics_, 378, 686–707.

#v(0.4em)
*[8]* Welch, G., & Bishop, G. (2006). _An Introduction to the Kalman Filter_. Technical Report TR 95-041, University of North Carolina at Chapel Hill, Department of Computer Science.
