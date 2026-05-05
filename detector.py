"""
detector.py
-----------
OpenCV-based detector for ANY round object in video.

Supports:
  - HSV colour thresholding (multiple colours tried simultaneously)
  - MOG2 background subtraction
  - Circularity filtering
  - Velocity-based outlier removal
  - Kinematic RANSAC Parabola filtering
  - Annotated video output
"""

import cv2
import numpy as np
import warnings
import torch
from config import COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Single-frame detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_frame(frame, color_ranges, min_r=3, max_r=80, bg_sub=None, last_pos=None):
    """
    Detect the most circular blob matching any of the given colours.
    Returns (cx, cy, radius) or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Combine masks for all candidate colours
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for cr in color_ranges:
        m = cv2.inRange(hsv, cr["lower"], cr["upper"])
        mask = cv2.bitwise_or(mask, m)

    # Foreground gating
    if bg_sub is not None:
        fg   = bg_sub.apply(frame)
        fg   = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(mask, fg)

    # Morphological cleanup
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best, best_score = None, -1
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if not (min_r <= r <= max_r):
            continue
            
        dist_score = 1.0
        if last_pos is not None:
            dist = np.hypot(cx - last_pos[0], cy - last_pos[1])
            if dist > max(200, 5 * max_r):
                continue
            dist_score = 1.0 / (1.0 + dist / 150.0)

        perim = cv2.arcLength(cnt, True)
        circ  = (4 * np.pi * area) / (perim**2 + 1e-6)
        
        score = circ * dist_score
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(r))

    return best if (best and best_score > 0.15) else None


# ─────────────────────────────────────────────────────────────────────────────
# Full video trajectory extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_trajectory(video_path, cfg,
                       color_override=None,
                       use_bg_sub=True,
                       max_frames=None,
                       annotate_output=None):
    """
    Process a video and return the detected object trajectory.

    Parameters
    ----------
    video_path      : str   – input video path
    cfg             : dict  – object config from config.py
    color_override  : list  – override colour keys
    use_bg_sub      : bool  – MOG2 background subtraction
    max_frames      : int   – max frames to process
    annotate_output : str   – if given, write annotated video here

    Returns
    -------
    dict:
      times, xs, ys : detected positions (pixel)
      fps, width, height
      frame_indices1
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames: total = min(total, max_frames)

    # --- ROI Selection ---
    ret, initial_frame = cap.read()
    color_ranges = []
    last_known_pos = None
    if ret:
        print("\n--- INSTRUCTIONS ---")
        print("1. A window will open with the first frame of your video.")
        print("2. Click and drag a rectangle around the reference object you want to track.")
        print("3. Press SPACE or ENTER to confirm.")
        print("--------------------\n")
        
        display_img = initial_frame.copy()
        scale = 1400.0 / float(max(W, 1))
        
        window_name = "Select Object to Track"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, int(W * scale), int(H * scale))
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        roi = cv2.selectROI(window_name, cv2.resize(display_img, (int(W * scale), int(H * scale))), showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
        
        if roi != (0, 0, 0, 0):
            x, y, box_w, box_h = [int(v / scale) for v in roi]
            cx, cy = x + box_w//2, y + box_h//2
            inner_w, inner_h = max(1, box_w//2), max(1, box_h//2)
            
            roi_crop = initial_frame[max(0, cy - inner_h//2) : min(H, cy + inner_h//2), max(0, cx - inner_w//2) : min(W, cx + inner_w//2)]
            if roi_crop.size > 0:
                hsv_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2HSV)
                h_med = np.median(hsv_roi[:,:,0])
                s_med = np.median(hsv_roi[:,:,1])
                v_med = np.median(hsv_roi[:,:,2])
                
                c_lower = np.array([max(0, int(h_med - 20)), max(0, int(s_med - 60)), max(0, int(v_med - 60))])
                c_upper = np.array([min(179, int(h_med + 20)), min(255, int(s_med + 60)), min(255, int(v_med + 60))])
                color_ranges = [{"lower": c_lower, "upper": c_upper}]
                last_known_pos = (cx, cy)
                print(f"[detector] Custom ROI Selected! Bounds: H:{int(h_med)}, S:{int(s_med)}, V:{int(v_med)}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    if not color_ranges:
        colors  = color_override or cfg["colors"]
        color_ranges = [COLORS[k] for k in colors if k in COLORS]
        print(f"[detector] Default config colours: {colors}")

    min_r   = cfg.get("min_radius_px", 3)
    max_r   = cfg.get("max_radius_px", 80)
    bg_sub  = cv2.createBackgroundSubtractorMOG2(
                  history=200, varThreshold=40, detectShadows=False
              ) if use_bg_sub else None

    writer = None
    if annotate_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotate_output, fourcc, fps, (W, H))

    # Initialize MiDaS Depth Model
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.eval().to(device)
    except Exception as e:
        print(f"[detector] Error loading MiDaS depth model: {e}")
        midas = None
        transform = None
        device = None

    times, xs, ys, zs, fids = [], [], [], [], []
    trail = []     # last N positions for trail drawing
    idx = 0
    initial_depth_val = None

    print(f"[detector] {cfg['display_name']}  |  "
          f"{W}x{H}@{fps:.0f}fps")

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and idx >= max_frames):
            break

        det = detect_frame(frame, color_ranges, min_r, max_r, bg_sub, last_known_pos)
        if det:
            cx, cy, r = det
            last_known_pos = (cx, cy)

            
            # --- Depth computation ---
            z_rel = 1.0
            if midas is not None and transform is not None:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_batch = transform(img_rgb).to(device)
                with torch.no_grad():
                    prediction = midas(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=img_rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth_map = prediction.cpu().numpy()
                
                tx = max(0, int(cx - r))
                ty = max(0, int(cy - r))
                tw = min(W - tx, int(r * 2))
                th = min(H - ty, int(r * 2))
                
                if tw > 0 and th > 0:
                    obj_depth = depth_map[ty:ty+th, tx:tx+tw]
                    current_inverse_depth = np.median(obj_depth)
                    if initial_depth_val is None and current_inverse_depth > 0:
                        initial_depth_val = current_inverse_depth
                    
                    if current_inverse_depth > 0 and initial_depth_val is not None:
                        z_rel = initial_depth_val / current_inverse_depth

            times.append(idx / fps)
            xs.append(cx); ys.append(cy)
            zs.append(z_rel)
            fids.append(idx)
            trail.append((int(cx), int(cy)))
            if len(trail) > 40: trail.pop(0)

            if writer:
                # Draw trail
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    cv2.line(frame, trail[i-1], trail[i],
                             (int(255*alpha), int(100*(1-alpha)), 50), 2)
                # Detection circle
                cv2.circle(frame, (int(cx), int(cy)), int(r), (0,255,0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 3,      (0,0,255), -1)
                
                # Annotate depth text
                text = f"{z_rel:.2f}x dist"
                cv2.putText(frame, text,
                            (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if writer:
            cv2.putText(frame, f"Frame {idx}  |  Det: {len(times)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,255), 1)
            writer.write(frame)

        idx += 1
        if idx % 200 == 0:
            pct = 100*len(times)/max(idx,1)
            print(f"  {idx}/{total}  |  detections: {len(times)} ({pct:.0f}%)")

    cap.release()
    if writer: writer.release()

    # Convert to numpy arrays
    t_raw = np.array(times)
    x_raw = np.array(xs)
    y_raw = np.array(ys)
    fids_raw = np.array(fids, dtype=int)
    z_raw = np.array(zs)

    # ---> ACTUALLY CALL THE CLEAN FUNCTION HERE <---
    if len(t_raw) > 4:
        t_clean, x_clean, y_clean = clean(t_raw, x_raw, y_raw)
        
        # We also need to keep frame_indices aligned with the cleaned data
        # by finding which timestamps survived the cleaning process.
        valid_indices = np.isin(t_raw, t_clean)
        fids_clean = fids_raw[valid_indices]
        z_clean = z_raw[valid_indices]
    else:
        t_clean, x_clean, y_clean, fids_clean, z_clean = t_raw, x_raw, y_raw, fids_raw, z_raw

    rate = 100 * len(t_clean) / max(idx, 1)
    print(f"[detector] Done  →  {len(t_clean)} clean positions kept from {len(t_raw)} raw "
          f"({rate:.1f}% final valid rate)")

    return {
        "times": t_clean, 
        "xs": x_clean, 
        "ys": y_clean,
        "zs": z_clean,
        "fps": fps, 
        "width": W, 
        "height": H,
        "frame_indices": fids_clean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pixel → metre conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_meters(xs, ys, height_px, scene_width_m, scale_override=None):
    """Convert pixel coordinates to metres."""
    if scale_override:
        scale = scale_override
    else:
        span  = max(xs.max() - xs.min(), 1.0)
        scale = span / scene_width_m          # px/m

    x_m = (xs - xs.min()) / scale
    y_m = (height_px - ys) / scale            # flip y
    y_m = y_m - y_m.min()
    return x_m, y_m


# ─────────────────────────────────────────────────────────────────────────────
# Outlier removal
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.neighbors import LocalOutlierFactor
import warnings
import numpy as np

def clean(t, x, y, ransac_iters=250, threshold_px=15.0, lof_neighbors=20):
    """LOF + velocity filter + Strict Kinematic RANSAC for parabolic paths."""
    
    if len(t) < 4:
        return t, x, y

    # 1. Local Outlier Factor (LOF) Spatial Filter
    # Intelligently drops sparse, scattered jitters based on local point density
    n_neighbors = min(lof_neighbors, len(t) - 1)
    if n_neighbors >= 2:
        X = np.column_stack((x, y))
        # 'auto' contamination uses an offset of -1.5 (similar to Isolation Forest)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        is_inlier = lof.fit_predict(X) == 1
        t_f, x_f, y_f = t[is_inlier], x[is_inlier], y[is_inlier]
    else:
        t_f, x_f, y_f = t, x, y

    if len(t_f) < 4:
        return t_f, x_f, y_f

    # 2. Velocity Filter 
    # Removes teleportation jitters along the path that LOF might consider dense
    dt  = np.diff(t_f) + 1e-9
    spd = np.sqrt(np.diff(x_f)**2 + np.diff(y_f)**2) / dt
    med = np.median(spd)
    ok  = np.concatenate([[True], spd < med * 10])
    t_f, x_f, y_f = t_f[ok], x_f[ok], y_f[ok]

    if len(t_f) < 4:
        return t_f, x_f, y_f

    # 3. Kinematic RANSAC Parabola Filter
    # Enforces the physical shape of the trajectory
    best_inliers = np.ones(len(t_f), dtype=bool)
    best_inlier_count = 0
    best_error = np.inf

    span = max(np.ptp(x_f), np.ptp(y_f))
    thresh = min(max(span * 0.05, 5.0), threshold_px) 

    with warnings.catch_warnings():
        warnings.simplefilter('ignore') 
        for _ in range(ransac_iters):
            idx = np.random.choice(len(t_f), 3, replace=False)
            t_s, x_s, y_s = t_f[idx], x_f[idx], y_f[idx]

            try:
                p_x = np.polyfit(t_s, x_s, 2)
                p_y = np.polyfit(t_s, y_s, 2)
            except np.linalg.LinAlgError:
                continue

            x_pred = np.polyval(p_x, t_f)
            y_pred = np.polyval(p_y, t_f)

            dist = np.sqrt((x_f - x_pred)**2 + (y_f - y_pred)**2)
            inliers = dist < thresh
            count = np.sum(inliers)

            if count > best_inlier_count:
                best_inlier_count = count
                best_inliers = inliers
                best_error = np.mean(dist[inliers])
            elif count == best_inlier_count:
                err = np.mean(dist[inliers])
                if err < best_error:
                    best_inliers = inliers
                    best_error = err

    # 4. Final Refit & Strict Thresholding
    if best_inlier_count >= 4:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p_x = np.polyfit(t_f[best_inliers], x_f[best_inliers], 2)
            p_y = np.polyfit(t_f[best_inliers], y_f[best_inliers], 2)

        x_pred = np.polyval(p_x, t_f)
        y_pred = np.polyval(p_y, t_f)
        dist = np.sqrt((x_f - x_pred)**2 + (y_f - y_pred)**2)

        median_err = np.median(dist[best_inliers])
        strict_inliers = dist < max(median_err * 3.0, 5.0)

        if np.sum(strict_inliers) >= 4:
            return t_f[strict_inliers], x_f[strict_inliers], y_f[strict_inliers]

    return t_f[best_inliers], x_f[best_inliers], y_f[best_inliers]