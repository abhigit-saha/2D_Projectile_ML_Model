#!/usr/bin/env python3
"""
Online 3-D Trajectory Estimation of a Flying Object from a Monocular Image Sequence
Implementation of Herrejon, Kagami, & Hashimoto (IEEE/RSJ IROS 2009)
Enhanced for real-world video tracking, arbitrary camera angles, and physical metrics.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters (Eq. 1)."""
    fu: float
    fv: float
    cu: float
    cv: float
    width: int = 640
    height: int = 480

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 65.0) -> "CameraIntrinsics":
        """Estimates intrinsics from resolution and horizontal field-of-view (typical smartphone/webcam ~65 deg)."""
        fu = (width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
        fv = fu
        cu = width / 2.0
        cv = height / 2.0
        return cls(fu=fu, fv=fv, cu=cu, cv=cv, width=width, height=height)


@dataclass
class TrajectoryMetrics:
    """Physical parameters derived from the 9 kinematic coefficients C_1..C_9."""
    C: np.ndarray                 # [C1, C2, C3, C4, C5, C6, C7, C8, C9]
    pos_0: np.ndarray             # Initial 3D position in camera frame (m)
    vel_0: np.ndarray             # Initial 3D velocity in camera frame (m/s)
    acc_cam: np.ndarray           # 3D acceleration vector in camera frame (m/s^2)
    speed_0: float                # Initial launch speed (m/s)
    launch_angle_deg: float       # Launch angle relative to the horizontal ground plane (deg)
    horizontal_speed: float       # Constant horizontal ground speed (m/s)
    vertical_speed_0: float       # Initial vertical speed against gravity (m/s)
    t_apex: float                 # Time to reach maximum height (s)
    apex_height_rel: float        # Maximum height gained above release point (m)
    t_flight_sym: float           # Flight time back to release height (s)
    horizontal_range_sym: float   # Horizontal distance covered at release height (m)
    R_cam_to_world: np.ndarray    # 3x3 rotation matrix aligning camera frame with gravity-aligned world frame


class HerrejonRLSEstimator:
    """
    Implements the Recursive Least Squares (RLS) 3D trajectory estimator
    from Equations (15)-(25) of Herrejon et al. (2009), plus automatic
    cheirality sign disambiguation and SVD homogeneous fallback for level cameras.
    """

    def __init__(self, intrinsics: CameraIntrinsics, g: float = 9.81, p_init: float = 1e6):
        self.cam = intrinsics
        self.g = g
        # State vector a_hat (8x1) and covariance matrix P (8x8) from Eq. (18), (23)-(25)
        self.a_hat = np.zeros((8, 1), dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * p_init

        # Store history for SVD fallback (handles horizontal camera singularity C9 -> 0)
        self.t_history: List[float] = []
        self.u_prime_history: List[float] = []
        self.v_prime_history: List[float] = []

    def update(self, t_i: float, u_prime: float, v_prime: float) -> Optional[np.ndarray]:
        """
        Ingests a single 2D pixel observation (u_prime, v_prime) at timestamp t_i (seconds)
        and updates the RLS state estimate (Eq. 16-25).
        Returns the 9-element vector C = [C1..C9] once >= 5 frames are observed.
        """
        self.t_history.append(t_i)
        self.u_prime_history.append(u_prime)
        self.v_prime_history.append(v_prime)

        # Eq. (7)-(8): Center coordinates relative to principal point (c_u, c_v)
        u_i = u_prime - self.cam.cu
        v_i = v_prime - self.cam.cv

        fu, fv = self.cam.fu, self.cam.fv
        t2 = t_i * t_i

        # Eq. (17): Observation matrix H_i (2x8)
        H_i = np.array([
            [fu, fu * t_i, fu * t2, 0.0, 0.0, 0.0, u_i, u_i * t_i],
            [0.0, 0.0, 0.0, fv, fv * t_i, fv * t2, v_i, v_i * t_i]
        ], dtype=np.float64)

        # Eq. (19): Observation vector q_i (2x1)
        q_i = np.array([
            [-u_i * t2],
            [-v_i * t2]
        ], dtype=np.float64)

        # Eq. (24)-(25): Covariance and Gain update using Joseph/Woodbury matrix inversion lemma
        # Numerically more stable equivalent of P_i = (P_{i-1}^{-1} + H_i^T H_i)^{-1}
        S_i = np.eye(2) + H_i @ self.P @ H_i.T
        K_i = self.P @ H_i.T @ np.linalg.inv(S_i)
        self.P = (np.eye(8) - K_i @ H_i) @ self.P

        # Eq. (23): State update
        innovation = q_i - (H_i @ self.a_hat)
        self.a_hat = self.a_hat + K_i @ innovation

        if len(self.t_history) < 5:
            return None

        return self.recover_C_parameters()

    def recover_C_parameters(self, use_svd_if_degenerate: bool = True) -> np.ndarray:
        """
        Recovers C = [C1..C9] from the 8-state RLS vector a_hat using Eq. (21)-(22)
        with Cheirality constraint (C7 = z0 > 0).
        If the camera is nearly horizontal (|a3| or |a6| >> 1, meaning C9 -> 0),
        automatically cross-checks with the unnormalized homogeneous SVD solver.
        """
        a = self.a_hat.flatten()
        a1, a2, a3, a4, a5, a6, a7, a8 = a

        # Eq. (22): Scale magnitude from gravity constraint C3^2 + C6^2 + C9^2 = g^2 / 4
        d_mag = (2.0 / self.g) * np.sqrt(a3**2 + a6**2 + 1.0)

        # Cheirality constraint: The ball starts in front of the camera lens (z(0) = C7 = a7/d > 0)
        sign_d = 1.0 if a7 >= 0 else -1.0
        d = sign_d * d_mag

        C_rls = np.array([
            a1 / d, a2 / d, a3 / d,
            a4 / d, a5 / d, a6 / d,
            a7 / d, a8 / d, 1.0 / d
        ], dtype=np.float64)

        # Check if camera optical axis is nearly perpendicular to gravity (C9 ~ 0)
        # In that regime, dividing by C9 amplifies noise; homogeneous SVD is exact for any angle.
        if use_svd_if_degenerate and len(self.t_history) >= 6:
            if abs(a3) > 15.0 or abs(a6) > 15.0:
                return self.solve_homogeneous_svd()

        return C_rls

    def solve_homogeneous_svd(self) -> np.ndarray:
        """
        Solves the exact same projective equations (11)-(12) without dividing by C9,
        eliminating the singularity when the camera is held horizontally.
        """
        N = len(self.t_history)
        M = np.zeros((2 * N, 9), dtype=np.float64)
        fu, fv = self.cam.fu, self.cam.fv

        for i in range(N):
            t_i = self.t_history[i]
            t2 = t_i * t_i
            u_i = self.u_prime_history[i] - self.cam.cu
            v_i = self.v_prime_history[i] - self.cam.cv

            M[2 * i] = [fu, fu * t_i, fu * t2, 0.0, 0.0, 0.0, u_i, u_i * t_i, u_i * t2]
            M[2 * i + 1] = [0.0, 0.0, 0.0, fv, fv * t_i, fv * t2, v_i, v_i * t_i, v_i * t2]

        # Right singular vector corresponding to smallest singular value
        _, _, Vt = np.linalg.svd(M)
        c_raw = Vt[-1, :]

        # Enforce gravity magnitude constraint: C3^2 + C6^2 + C9^2 = g^2 / 4
        acc_norm = np.sqrt(c_raw[2]**2 + c_raw[5]**2 + c_raw[8]**2)
        if acc_norm < 1e-12:
            return c_raw
        scale = (self.g / 2.0) / acc_norm
        C = c_raw * scale

        # Enforce Cheirality constraint: C7 = z(0) > 0
        if C[6] < 0:
            C = -C
        return C

    def evaluate_3d_position(self, C: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluates 3D camera coordinates (x(t), y(t), z(t)) at time array t (Eq. 4-6)."""
        x = C[0] + C[1] * t + C[2] * (t ** 2)
        y = C[3] + C[4] * t + C[5] * (t ** 2)
        z = C[6] + C[7] * t + C[8] * (t ** 2)
        return x, y, z

    def project_to_image(self, C: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects 3D trajectory back onto 2D image coordinates (u', v') using Eq. (7)-(8)."""
        x, y, z = self.evaluate_3d_position(C, t)
        # Prevent division by zero for points behind camera
        z_safe = np.where(np.abs(z) < 1e-5, 1e-5, z)
        u_prime = -self.cam.fu * (x / z_safe) + self.cam.cu
        v_prime = -self.cam.fv * (y / z_safe) + self.cam.cv
        return u_prime, v_prime

    def compute_physical_metrics(self, C: np.ndarray, ground_drop_m: float = 0.0) -> TrajectoryMetrics:
        """
        Extracts full real-world kinematics (velocity, launch angle, range, apex height,
        flight time) by aligning the recovered gravity vector [2*C3, 2*C6, 2*C9].
        """
        pos_0 = np.array([C[0], C[3], C[6]])
        vel_0 = np.array([C[1], C[4], C[7]])
        acc_cam = 2.0 * np.array([C[2], C[5], C[8]])  # True acceleration vector a = 2 * [C3, C6, C9]

        speed_0 = float(np.linalg.norm(vel_0))

        # Unit vector pointing straight UP against gravity in camera coordinates
        g_norm = np.linalg.norm(acc_cam)
        up_unit = -acc_cam / (g_norm if g_norm > 1e-9 else 1.0)

        # Decompose initial velocity into vertical (along up_unit) and horizontal components
        v_vert_0 = float(np.dot(vel_0, up_unit))
        v_horiz_vec = vel_0 - v_vert_0 * up_unit
        v_horiz = float(np.linalg.norm(v_horiz_vec))

        if v_horiz > 1e-6:
            horiz_unit = v_horiz_vec / v_horiz
        else:
            # Arbitrary orthogonal vector if thrown straight up
            arbitrary = np.array([1.0, 0.0, 0.0]) if abs(up_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            horiz_unit = np.cross(up_unit, arbitrary)
            horiz_unit /= np.linalg.norm(horiz_unit)

        lateral_unit = np.cross(up_unit, horiz_unit)
        # Rotation matrix from camera coordinates to gravity-aligned world coordinates:
        # X_world = horizontal direction of throw, Y_world = lateral (0 for planar parabola), Z_world = straight up
        R_cam_to_world = np.vstack([horiz_unit, lateral_unit, up_unit])

        launch_angle_deg = float(np.degrees(np.arctan2(v_vert_0, v_horiz)))
        t_apex = max(0.0, v_vert_0 / self.g)
        apex_height_rel = (v_vert_0 ** 2) / (2.0 * self.g) if v_vert_0 > 0 else 0.0

        # Flight time until the ball lands at height = -ground_drop_m relative to release point
        discriminant = v_vert_0**2 + 2.0 * self.g * ground_drop_m
        if discriminant >= 0:
            t_flight = (v_vert_0 + np.sqrt(discriminant)) / self.g
        else:
            t_flight = 2.0 * max(0.0, t_apex)

        t_flight = max(0.0, float(t_flight))
        horiz_range = v_horiz * t_flight

        return TrajectoryMetrics(
            C=C,
            pos_0=pos_0,
            vel_0=vel_0,
            acc_cam=acc_cam,
            speed_0=speed_0,
            launch_angle_deg=launch_angle_deg,
            horizontal_speed=v_horiz,
            vertical_speed_0=v_vert_0,
            t_apex=t_apex,
            apex_height_rel=apex_height_rel,
            t_flight_sym=t_flight,
            horizontal_range_sym=horiz_range,
            R_cam_to_world=R_cam_to_world,
        )


# ==============================================================================
# PART 2: COMPUTER VISION FRONTEND (2D BALL DETECTION IN REAL VIDEOS)
# ==============================================================================

class BallDetector2D:
    """
    Extracts the 2D centroid (u', v') of a thrown ball from video frames.
    Supports:
      1. 'motion': MOG2 background subtraction + circularity/area filtering (works on any ball color)
      2. 'hsv': HSV color thresholding (best for bright tennis/nerf/colored balls)
      3. 'manual': Interactive user click on each frame for ground-truth testing
    """

    def __init__(
        self,
        mode: str = "motion",
        hsv_lower: Tuple[int, int, int] = (25, 70, 70),
        hsv_upper: Tuple[int, int, int] = (65, 255, 255),
        min_radius_px: float = 3.0,
        max_radius_px: float = 120.0,
    ):
        self.mode = mode
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_radius_px = min_radius_px
        self.max_radius_px = max_radius_px
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=32, detectShadows=False
        )
        self.last_pos: Optional[Tuple[float, float]] = None

    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Returns (u_prime, v_prime, radius_px) if the ball is detected, else None."""
        if self.mode == "manual":
            return self._detect_manual(frame)
        elif self.mode == "hsv":
            mask = self._mask_hsv(frame)
        else:
            mask = self._mask_motion(frame)

        return self._find_best_circular_blob(mask)

    def _mask_hsv(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _mask_motion(self, frame: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = self.bg_subtractor.apply(blurred)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        return fg_mask

    def _find_best_circular_blob(self, mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = None
        best_score = -1e9

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * (self.min_radius_px ** 2):
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            # Circularity = 4*pi*Area / Perimeter^2 (1.0 for a perfect circle)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.35:
                continue

            (u, v), radius = cv2.minEnclosingCircle(cnt)
            if not (self.min_radius_px <= radius <= self.max_radius_px):
                continue

            # Moments for sub-pixel centroid accuracy
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                u = float(M["m10"] / M["m00"])
                v = float(M["m01"] / M["m00"])

            # Prefer high circularity and proximity to previous detection
            prox_penalty = 0.0
            if self.last_pos is not None:
                dist = np.hypot(u - self.last_pos[0], v - self.last_pos[1])
                prox_penalty = 0.005 * dist

            score = circularity - prox_penalty
            if score > best_score:
                best_score = score
                best_candidate = (float(u), float(v), float(radius))

        if best_candidate is not None:
            self.last_pos = (best_candidate[0], best_candidate[1])
        return best_candidate

    def _detect_manual(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        clicked_pt = []

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_pt.append((float(x), float(y)))

        disp = frame.copy()
        cv2.putText(
            disp,
            "Click ball center | SPACE: skip frame | Q: finish",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        win_name = "Manual Ball Tracker"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, on_mouse)

        while True:
            cv2.imshow(win_name, disp)
            key = cv2.waitKey(20) & 0xFF
            if clicked_pt:
                return (clicked_pt[0][0], clicked_pt[0][1], 8.0)
            if key == ord(" "):
                return None
            if key == ord("q"):
                cv2.destroyWindow(win_name)
                sys.exit(0)


# ==============================================================================
# PART 3: PAPER SIMULATION BENCHMARK (SECTION IV-A VERIFICATION)
# ==============================================================================

def run_paper_verification():
    """
    Reproduces Camera Position 1 and Camera Position 2 from Section IV-A of
    Herrejon et al. (2009) across noise levels sigma = 0, 0.5, 1.0, 2.0 pixels.
    """
    print("=" * 78)
    print("VERIFYING IMPLEMENTATION AGAINST HERREJON ET AL. (2009) - SECTION IV-A")
    print("=" * 78)

    configs = [
        {
            "name": "Camera Position 1 (Eq. 31-33, Receding/Approaching Oblique)",
            "intrinsics": CameraIntrinsics(fu=799.0, fv=799.0, cu=267.2, cv=205.7, width=640, height=480),
            "C_true": np.array([-0.599, 1.374, 0.137, 0.317, -0.299, 0.030, 2.091, -4.356, 4.898]),
        },
        {
            "name": "Camera Position 2 (Eq. 34-36, Negative C9 / Tilted Frame)",
            "intrinsics": CameraIntrinsics(fu=550.0, fv=550.0, cu=267.0, cv=205.0, width=640, height=480),
            "C_true": np.array([0.212, 0.427, -0.623, -0.314, 1.374, 0.158, 0.695, 4.345, -4.857]),
        },
    ]

    # 60 frames over 0.87 seconds (~69 Hz sampling rate from Section IV-A)
    num_frames = 60
    t_seq = np.linspace(0.0, 0.87, num_frames)
    noise_levels = [0.0, 0.5, 1.0, 2.0]

    for cfg in configs:
        print(f"\n---> {cfg['name']}")
        C_true = cfg["C_true"]
        cam = cfg["intrinsics"]
        # Exact gravity magnitude used by the synthetic coefficients in the paper
        g_sim = 2.0 * np.sqrt(C_true[2]**2 + C_true[5]**2 + C_true[8]**2)

        print(f"Ground Truth C : {np.round(C_true, 3)}")

        for sigma in noise_levels:
            np.random.seed(42)
            estimator = HerrejonRLSEstimator(cam, g=g_sim)

            for t_i in t_seq:
                x_i = C_true[0] + C_true[1] * t_i + C_true[2] * (t_i ** 2)
                y_i = C_true[3] + C_true[4] * t_i + C_true[5] * (t_i ** 2)
                z_i = C_true[6] + C_true[7] * t_i + C_true[8] * (t_i ** 2)

                u_p = -cam.fu * (x_i / z_i) + cam.cu + np.random.normal(0, sigma)
                v_p = -cam.fv * (y_i / z_i) + cam.cv + np.random.normal(0, sigma)

                C_est = estimator.update(t_i, u_p, v_p)

            # Compute trajectory RMSE across the sequence
            x_true, y_true, z_true = estimator.evaluate_3d_position(C_true, t_seq)
            x_fit, y_fit, z_fit = estimator.evaluate_3d_position(C_est, t_seq)
            rmse = np.sqrt(np.mean((x_true - x_fit)**2 + (y_true - y_fit)**2 + (z_true - z_fit)**2))

            print(f"Noise={sigma:3.1f} px | Est C: {np.round(C_est, 3)} | 3D RMSE: {rmse:.4f} m")

        # Display extracted physical metrics for noiseless run
        metrics = estimator.compute_physical_metrics(C_true)
        print(
            f"Physical Metrics -> Launch Speed: {metrics.speed_0:.2f} m/s | "
            f"Launch Angle: {metrics.launch_angle_deg:.1f} deg | "
            f"Apex Gain: {metrics.apex_height_rel:.2f} m | "
            f"Horizontal Range: {metrics.horizontal_range_sym:.2f} m"
        )


# ==============================================================================
# PART 4: REAL-WORLD VIDEO PIPELINE & 3D VISUALIZATION
# ==============================================================================

def process_video(
    video_path: str,
    mode: str = "motion",
    hfov_deg: float = 65.0,
    fu_override: Optional[float] = None,
    hsv_lower: Tuple[int, int, int] = (25, 70, 70),
    hsv_upper: Tuple[int, int, int] = (65, 255, 255),
    ground_drop_m: float = 0.0,
    output_video_path: Optional[str] = "trajectory_output.mp4",
    show_live: bool = True,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 60.0

    if fu_override is not None:
        cam = CameraIntrinsics(fu=fu_override, fv=fu_override, cu=width / 2.0, cv=height / 2.0, width=width, height=height)
    else:
        cam = CameraIntrinsics.from_fov(width, height, hfov_deg=hfov_deg)

    print(f"[INFO] Video: {width}x{height} @ {fps:.2f} FPS")
    print(f"[INFO] Camera Intrinsics: fu={cam.fu:.1f}, fv={cam.fv:.1f}, cu={cam.cu:.1f}, cv={cam.cv:.1f}")

    detector = BallDetector2D(mode=mode, hsv_lower=hsv_lower, hsv_upper=hsv_upper)
    estimator = HerrejonRLSEstimator(cam, g=9.81)

    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_idx = 0
    t0_FirstDetection: Optional[float] = None
    latest_C: Optional[np.ndarray] = None
    latest_metrics: Optional[TrajectoryMetrics] = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_video = frame_idx / fps
        detection = detector.detect(frame)
        vis = frame.copy()

        if detection is not None:
            u_p, v_p, radius = detection
            if t0_FirstDetection is None:
                t0_FirstDetection = t_video

            t_rel = t_video - t0_FirstDetection
            latest_C = estimator.update(t_rel, u_p, v_p)

            # Draw current detected ball
            cv2.circle(vis, (int(u_p), int(v_p)), int(max(radius, 5)), (0, 255, 0), 2)
            cv2.circle(vis, (int(u_p), int(v_p)), 3, (0, 0, 255), -1)

        # Draw historical 2D observations
        for u_hist, v_hist in zip(estimator.u_prime_history, estimator.v_prime_history):
            cv2.circle(vis, (int(u_hist), int(v_hist)), 3, (0, 180, 255), -1)

        # Once RLS has >= 6 observations, project predicted 3D trajectory forward in time
        if latest_C is not None and len(estimator.t_history) >= 6:
            latest_metrics = estimator.compute_physical_metrics(latest_C, ground_drop_m=ground_drop_m)
            t_max_pred = max(estimator.t_history[-1] + 0.5, latest_metrics.t_flight_sym)
            t_pred = np.linspace(0.0, min(t_max_pred, 3.0), 120)

            u_fit, v_fit = estimator.project_to_image(latest_C, t_pred)
            _, _, z_fit = estimator.evaluate_3d_position(latest_C, t_pred)

            pts_2d = []
            for uf, vf, zf in zip(u_fit, v_fit, z_fit):
                if zf > 0.1 and -width <= uf <= 2 * width and -height <= vf <= 2 * height:
                    pts_2d.append([int(uf), int(vf)])

            if len(pts_2d) >= 2:
                pts_arr = np.array(pts_2d, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts_arr], isClosed=False, color=(255, 80, 0), thickness=2)

            # Overlay telemetry HUD
            x_cur, y_cur, z_cur = estimator.evaluate_3d_position(
                latest_C, np.array([estimator.t_history[-1]])
            )
            hud_lines = [
                f"Obs Frames: {len(estimator.t_history)}",
                f"Current 3D Pos (Cam): X={x_cur[0]:.2f}m, Y={y_cur[0]:.2f}m, Z(depth)={z_cur[0]:.2f}m",
                f"Launch Speed: {latest_metrics.speed_0:.2f} m/s ({latest_metrics.speed_0 * 3.6:.1f} km/h)",
                f"Launch Angle (World): {latest_metrics.launch_angle_deg:.1f} deg",
                f"Apex Height Gain: {latest_metrics.apex_height_rel:.2f} m (at t={latest_metrics.t_apex:.2f}s)",
                f"Pred Horiz Distance: {latest_metrics.horizontal_range_sym:.2f} m (Flight={latest_metrics.t_flight_sym:.2f}s)",
            ]
            for idx, text in enumerate(hud_lines):
                cv2.putText(
                    vis, text, (15, 28 + idx * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA
                )
                cv2.putText(
                    vis, text, (15, 28 + idx * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (50, 255, 50), 1, cv2.LINE_AA
                )

        if writer is not None:
            writer.write(vis)

        if show_live:
            cv2.imshow("Monocular 3D Projectile Trajectory Estimator", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Final Summary & 3D Plot
    if latest_C is not None and latest_metrics is not None:
        print("\n" + "=" * 78)
        print("FINAL ESTIMATED 3D TRAJECTORY EQUATIONS & KINEMATIC SUMMARY")
        print("=" * 78)
        C = latest_C
        print(f"x_cam(t) = {C[0]:+.4f} {C[1]:+.4f}*t {C[2]:+.4f}*t^2  (m)")
        print(f"y_cam(t) = {C[3]:+.4f} {C[4]:+.4f}*t {C[5]:+.4f}*t^2  (m)")
        print(f"z_cam(t) = {C[6]:+.4f} {C[7]:+.4f}*t {C[8]:+.4f}*t^2  (m)")
        print("-" * 78)
        print(f"Initial Depth from Camera z(0) : {C[6]:.3f} m")
        print(f"Initial Speed |v0|             : {latest_metrics.speed_0:.3f} m/s ({latest_metrics.speed_0*3.6:.2f} km/h)")
        print(f"Horizontal Ground Speed        : {latest_metrics.horizontal_speed:.3f} m/s")
        print(f"Vertical Launch Speed          : {latest_metrics.vertical_speed_0:.3f} m/s")
        print(f"World Launch Angle             : {latest_metrics.launch_angle_deg:.2f} deg")
        print(f"Maximum Apex Height Gain       : {latest_metrics.apex_height_rel:.3f} m (at t = {latest_metrics.t_apex:.3f} s)")
        print(f"Predicted Flight Duration      : {latest_metrics.t_flight_sym:.3f} s")
        print(f"Predicted Horizontal Range     : {latest_metrics.horizontal_range_sym:.3f} m")
        print("=" * 78)

        plot_3d_reconstruction(estimator, latest_C, latest_metrics)
    else:
        print("[WARN] Insufficient ball detections (< 6 frames) to reconstruct 3D trajectory.")


def plot_3d_reconstruction(
    estimator: HerrejonRLSEstimator, C: np.ndarray, metrics: TrajectoryMetrics
):
    """Generates a 2-panel 3D plot: (1) Camera Coordinate Frame, (2) Gravity-Aligned World Frame."""
    t_obs = np.array(estimator.t_history)
    t_full = np.linspace(0.0, max(t_obs[-1], metrics.t_flight_sym), 100)

    x_obs, y_obs, z_obs = estimator.evaluate_3d_position(C, t_obs)
    x_full, y_full, z_full = estimator.evaluate_3d_position(C, t_full)

    # Transform to Gravity-Aligned World Frame (Origin at release point, Z_world straight up)
    pts_cam_full = np.vstack([x_full - C[0], y_full - C[3], z_full - C[6]])
    pts_world_full = metrics.R_cam_to_world @ pts_cam_full

    pts_cam_obs = np.vstack([x_obs - C[0], y_obs - C[3], z_obs - C[6]])
    pts_world_obs = metrics.R_cam_to_world @ pts_cam_obs

    fig = plt.figure(figsize=(14, 6))

    # Subplot 1: Camera Frame
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(x_full, z_full, y_full, "b--", label="Predicted 3D Path")
    ax1.scatter(x_obs, z_obs, y_obs, c="r", s=25, label="Observed Frames")
    ax1.scatter([0], [0], [0], c="k", marker="^", s=80, label="Camera Origin (0,0,0)")
    ax1.set_xlabel("Camera X (m)")
    ax1.set_ylabel("Camera Depth Z (m)")
    ax1.set_zlabel("Camera Y (m)")
    ax1.set_title("3D Trajectory in Camera Coordinates")
    ax1.legend()

    # Subplot 2: Gravity-Aligned World Frame
    ax2 = fig.add_subplot(122)
    ax2.plot(pts_world_full[0], pts_world_full[2], "b-", linewidth=2, label="Predicted World Parabola")
    ax2.scatter(pts_world_obs[0], pts_world_obs[2], c="r", s=30, zorder=5, label="Observed Frames")
    ax2.axhline(0, color="gray", linestyle=":", label="Release Height Plane")
    ax2.set_xlabel("Horizontal Distance Covered (m)")
    ax2.set_ylabel("Vertical Height Relative to Release (m)")
    ax2.set_title(
        f"Gravity-Aligned World Trajectory\n"
        f"Speed={metrics.speed_0:.2f} m/s | Angle={metrics.launch_angle_deg:.1f}° | Range={metrics.horizontal_range_sym:.2f} m"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("trajectory_3d_analysis.png", dpi=150)
    print("[INFO] Saved 3D trajectory plot to 'trajectory_3d_analysis.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Monocular 3D Projectile Trajectory Estimator (Herrejon et al., 2009)"
    )
    parser.add_argument(
        "--verify-paper",
        action="store_true",
        help="Run synthetic benchmark from Section IV-A of the paper to verify math.",
    )
    parser.add_argument("--video", type=str, default=None, help="Path to input video file.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["motion", "hsv", "manual"],
        default="motion",
        help="2D ball detection method: 'motion' (MOG2), 'hsv' (color mask), or 'manual' (click frame-by-frame).",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=65.0,
        help="Camera horizontal field of view in degrees (used if --fu is not provided).",
    )
    parser.add_argument(
        "--fu",
        type=float,
        default=None,
        help="Calibrated focal length in pixels (overrides --hfov).",
    )
    parser.add_argument(
        "--ground-drop",
        type=float,
        default=0.0,
        help="Vertical drop (in meters) from release point to ground level for total range calculation.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live OpenCV window display.",
    )
    args = parser.parse_args()

    if args.verify_paper or args.video is None:
        run_paper_verification()
        if args.video is None:
            print("\nTip: To run on your own video, execute:")
            print("  python3 monocular_projectile_3d.py --video throw.mp4 --mode motion --ground-drop 1.2")
            return

    process_video(
        video_path=args.video,
        mode=args.mode,
        hfov_deg=args.hfov,
        fu_override=args.fu,
        ground_drop_m=args.ground_drop,
        show_live=not args.no_display,
    )


if __name__ == "__main__":
    main()