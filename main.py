"""
main.py
-------
CARS Project: Prediction of Projectile Trajectory and Velocity
             in 2D through Machine Learning and Computer Vision

Sponsor : HEMRL, Pune
PI      : Dr. Anubhav Rawat, Dr. Ashutosh Mishra, Dr. A.K. Tiwari

USAGE
-----
  # List all supported objects
  python main.py --list

  # Simulate any object
  python main.py --simulate --object football --preset banana

  # Analyse a real video
  python main.py --video clip.mp4 --object tennis

  # Run all objects (demo/report)
  python main.py --demo
"""

import argparse, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

from config     import OBJECTS, get_object, list_objects
from physics    import simulate, KalmanTracker
from detector   import extract_trajectory, to_meters, clean
from models     import ProjectilePINN, LSTMPredictor, predict_all
from visualiser import (plot_comparison, plot_metrics, plot_velocity,
                        make_full_report, annotate_video)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CARS: 2-D Projectile Trajectory Prediction via ML + CV",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--object",      default="table_tennis",
                   help="Object type (use --list to see all)")
    p.add_argument("--preset",      default=None,
                   help="Simulation preset (e.g. topspin, banana, fastball)")
    p.add_argument("--video",       default=None,
                   help="Path to input video file")
    p.add_argument("--color",       default=None,
                   help="Override detection colour: white/orange/yellow/green/red")
    p.add_argument("--output",      default="results",
                   help="Output folder  (default: results/)")
    p.add_argument("--simulate",    action="store_true",
                   help="Use synthetic simulation instead of real video")
    p.add_argument("--demo",        action="store_true",
                   help="Run all objects and produce comparison report")
    p.add_argument("--list",        action="store_true",
                   help="List all available objects and exit")
    p.add_argument("--no-bg-sub",   action="store_true")
    p.add_argument("--max-frames",  type=int, default=None)
    p.add_argument("--pinn-iters",  type=int, default=5000)
    p.add_argument("--lstm-epochs", type=int, default=500)
    p.add_argument("--noise",       type=float, default=0.005,
                   help="Gaussian noise std (metres) for simulation")
    p.add_argument("--annotate-video", action="store_true",
                   help="Write annotated output video (real video mode only)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sim(cfg, preset_name, noise=0.005):
    key    = preset_name or list(cfg["presets"].keys())[0]
    params = cfg["presets"][key]
    print(f"\n[data] Simulating {cfg['display_name']}  ·  preset='{key}'")
    t, x, y, vx, vy = simulate(cfg, noise_m=noise, **params)

    # Sparse subsample (mimics tracking → sparse detections)
    rng = np.random.default_rng(42)
    n   = max(10, min(30, len(t)//4))
    idx = np.sort(rng.choice(len(t), n, replace=False))
    print(f"[data] Full traj: {len(t)} pts  |  sparse obs: {len(idx)}")
    return (t[idx], x[idx], y[idx],
            {"full": (t, x, y), "preset": key, "simulated": True})


def load_video(cfg, args):
    if not os.path.isfile(args.video):
        print(f"ERROR: not found: {args.video}"); sys.exit(1)

    colors = [args.color] if args.color else cfg["colors"]
    det    = extract_trajectory(
        args.video, cfg,
        color_override=colors,
        use_bg_sub=not args.no_bg_sub,
        max_frames=args.max_frames,
        annotate_output=None,
    )
    if len(det["times"]) < 5:
        print("ERROR: <5 detections. Try --color or --no-bg-sub"); sys.exit(1)

    x_m, y_m = to_meters(det["xs"], det["ys"], det["height"],
                          cfg["scene_width_m"])
    t, x, y  = clean(det["times"], x_m, y_m)
    print(f"[data] {len(t)} clean positions")
    return t, x, y, {"det": det, "simulated": False}


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg, t_obs, x_obs, y_obs, meta, args, out_dir=None):
    out = out_dir or args.output
    os.makedirs(out, exist_ok=True)

    # Ground truth (simulation only)
    if meta.get("simulated") and "full" in meta:
        t_gt, x_gt, y_gt = meta["full"]
    else:
        t_gt, x_gt, y_gt = t_obs, x_obs, y_obs

    print(f"\n{'='*60}")
    print(f"  CARS PIPELINE  —  {cfg['display_name']}")
    print(f"{'='*60}")

    # ── Train all models ──────────────────────────────────────────────────────
    results = predict_all(
        cfg, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
        pinn_iters=args.pinn_iters,
        lstm_epochs=args.lstm_epochs,
        verbose=True,
    )

    # ── Stroke classification ─────────────────────────────────────────────────
    stroke_label = results["pinn"]["label"]
    kin          = results["pinn"]["kin"]

    # ── Print metrics table ───────────────────────────────────────────────────
    print(f"\n  STROKE / PHASE  : {stroke_label}")
    print(f"  Speed V₀        : {kin['speed']:.2f} m/s")
    if cfg["has_spin"]:
        print(f"  Spin            : {kin['spin_rps']:.2f} rps")
    print(f"  Drag Cᴅ         : {kin['CD']:.3f}")
    print(f"\n  {'Model':<20} {'RMSE_x':>8} {'RMSE_y':>8} "
          f"{'ADE':>8} {'FDE':>8}")
    print(f"  {'-'*55}")
    for key, res in results.items():
        m = res.get("metrics", {})
        if m:
            print(f"  {key.upper():<20} "
                  f"{m.get('RMSE_x',np.nan):>8.4f} "
                  f"{m.get('RMSE_y',np.nan):>8.4f} "
                  f"{m.get('ADE',np.nan):>8.4f} "
                  f"{m.get('FDE',np.nan):>8.4f}")
    print(f"{'='*60}")

    # ── Save figures ──────────────────────────────────────────────────────────
    title = f"{cfg['display_name']}  ·  {stroke_label}"
    make_full_report(results, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
                     cfg, stroke_label,
                     save_path=os.path.join(out, "full_report.png"))

    plot_comparison(results, t_obs, x_obs, y_obs, t_gt, x_gt, y_gt,
                    title=title,
                    save_path=os.path.join(out, "comparison.png"))

    plot_metrics(results,
                 save_path=os.path.join(out, "metrics.png"))

    plot_velocity(t_obs, x_obs, y_obs, kin=kin,
                  save_path=os.path.join(out, "velocity.png"))

    print(f"\n  Saved to: {os.path.abspath(out)}/")

    # ── Optional annotated video ──────────────────────────────────────────────
    if args.annotate_video and not meta.get("simulated") and "det" in meta:
        det  = meta["det"]
        span = max(det["xs"].max()-det["xs"].min(), 1.)
        scl  = span / cfg["scene_width_m"]
        pinn_model  = results["pinn"]["model"]
        # Re-build KF for video annotation
        dt_v = 1./det["fps"]
        kf_v = KalmanTracker(dt=dt_v)
        for xi, yi in zip(det["xs"], det["ys"]):
            xm = (xi-det["xs"].min())/scl
            ym = (det["height"]-yi)/scl
            kf_v.update(xm, ym)
        annotate_video(
            args.video,
            os.path.join(out, "annotated.mp4"),
            cfg, det, pinn_model, None, kf_v,
            scale_pxm=scl, height_px=det["height"],
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode — all objects
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(args):
    print("\n" + "="*60)
    print("  CARS DEMO  —  All Objects Comparison")
    print("="*60)

    summary = []
    for key, cfg in OBJECTS.items():
        out = os.path.join(args.output, key)
        t, x, y, meta = load_sim(cfg, preset_name=None, noise=args.noise)
        quick = argparse.Namespace(
            pinn_iters=2000, lstm_epochs=300,
            annotate_video=False
        )
        res = run(cfg, t, x, y, meta, quick, out_dir=out)
        kin = res["pinn"]["kin"]
        summary.append({
            "name":  cfg["display_name"],
            "V0":    kin["speed"],
            "CD":    kin["CD"],
            "ADE":   res["pinn"]["metrics"]["ADE"],
            "label": res["pinn"]["label"],
        })

    print("\n" + "="*60)
    print("  SUMMARY — All Objects")
    print(f"  {'Object':<25} {'V0 m/s':>8} {'CD':>6} {'ADE m':>8}  Label")
    print("  " + "-"*60)
    for s in summary:
        print(f"  {s['name']:<25} {s['V0']:>8.2f} {s['CD']:>6.3f} "
              f"{s['ADE']:>8.4f}  {s['label']}")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.list:
        list_objects(); return

    print("\n" + "="*60)
    print("  CARS — Projectile Trajectory Prediction")
    print("  ML + Computer Vision  |  HEMRL / MNNIT Allahabad")
    print("="*60)

    if args.demo:
        run_demo(args); return

    cfg = get_object(args.object)

    if args.simulate or args.video is None:
        t, x, y, meta = load_sim(cfg, args.preset, noise=args.noise)
    else:
        t, x, y, meta = load_video(cfg, args)

    run(cfg, t, x, y, meta, args)
    print("\nDone! All outputs saved to:", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
