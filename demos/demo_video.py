import sys
import os
import json
import argparse
import urllib.request
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as tfhub
import imageio
import numpy as np
from tqdm import tqdm

from scipy.signal import butter, filtfilt

import cameralib
import poseviz


def _parse_float_list(s):
    """Parse '1.75' or '1.60,1.65,1.70' to a list of floats."""
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='MeTRAbs 3D pose estimation from video')
    parser.add_argument('video', help='Path to video file or URL')
    parser.add_argument('--ik', action='store_true', help='Run OpenSim Inverse Kinematics after pose estimation')
    parser.add_argument('--multi_person', action='store_true', help='Track and export all persons separately')
    parser.add_argument('--stationary', action='store_true', help='Fix horizontal drift (person stays centered, useful for any stationary-camera movement)')
    parser.add_argument('--combined_trc', action='store_true', help='In multi-person mode, also output a single combined TRC with all persons (markers suffixed p0_, p1_, ...)')
    parser.add_argument('--min_track_seconds', type=float, default=2.0, help='Minimum track duration in seconds for multi-person mode (default: 2.0)')
    parser.add_argument('--mass', type=float, default=69,
                        help='Subject mass in kg (single-person mode). Default: 69')
    parser.add_argument('--height', type=float, default=1.75,
                        help='Subject height in meters (single-person mode). Default: 1.75')
    parser.add_argument('--person_masses', type=str, default=None,
                        help='Comma-separated masses in kg for multi-person mode, sorted left-to-right on first frame (e.g. "60,65,70,75,80,85")')
    parser.add_argument('--person_heights', type=str, default=None,
                        help='Comma-separated heights in meters for multi-person mode, sorted left-to-right on first frame (e.g. "1.62,1.64,1.61,1.70,1.78,1.75")')
    args = parser.parse_args()

    USE_POSEVIZ = False

    model = tfhub.load('https://bit.ly/metrabs_l')
    skeleton = 'bml_movi_87'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    video_filepath = get_video(args.video)

    reader = imageio.get_reader(video_filepath, 'ffmpeg')
    fps = reader.get_meta_data().get('fps', 30.0)
    imshape = reader.get_data(0).shape[:2]

    try:
        total_batches = int(np.ceil(reader.count_frames() / 8))
    except:
        total_batches = None

    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)

    def frame_generator():
        for frame in imageio.get_reader(video_filepath, 'ffmpeg'):
            yield frame

    frame_batches = tf.data.Dataset.from_generator(
        frame_generator,
        output_signature=tf.TensorSpec(shape=(imshape[0], imshape[1], 3), dtype=tf.uint8)
    ).batch(8).prefetch(1)

    if USE_POSEVIZ:
        viz = poseviz.PoseViz(joint_names, joint_edges)

    # ── Multi-person tracking setup ──────────────────────────────────
    if args.multi_person:
        import supervision as sv
        tracker = sv.ByteTrack(
            track_activation_threshold=0.1,
            lost_track_buffer=60,
            minimum_matching_threshold=0.3,
            frame_rate=int(fps),
            minimum_consecutive_frames=1,
        )
        tracks = {}  # person_id -> {frames, poses3d, poses2d, confidences}
        frame_idx = 0
        _dbg_total_detections = 0
        _dbg_total_tracked = 0
        _dbg_empty_frames = 0
    else:
        all_poses3d = []
        all_poses2d = []
        all_confidences = []

    # ── Inference loop ───────────────────────────────────────────────
    for i, frame_batch in enumerate(tqdm(frame_batches, total=total_batches, desc="Inférence 3D en cours", unit="batch")):
        pred = model.detect_poses_batched(
            frame_batch, intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
            skeleton=skeleton)

        for frame, boxes, poses3d, poses2d in zip(
                frame_batch, pred['boxes'], pred['poses3d'], pred['poses2d']):
            boxes_np = boxes.numpy()
            poses3d_np = poses3d.numpy()
            poses2d_np = poses2d.numpy()

            # Convert boxes from [x, y, w, h, conf] to [x1, y1, x2, y2, conf]
            if len(boxes_np) > 0:
                boxes_np = boxes_np.copy()
                boxes_np[:, 2] = boxes_np[:, 0] + boxes_np[:, 2]  # x2 = x + w
                boxes_np[:, 3] = boxes_np[:, 1] + boxes_np[:, 3]  # y2 = y + h

            if USE_POSEVIZ:
                viz.update(frame=frame, boxes=boxes, poses=poses3d, camera=camera)

            if args.multi_person:
                # Track all persons
                if len(boxes_np) > 0:
                    _dbg_total_detections += len(boxes_np)
                    detections = sv.Detections(
                        xyxy=boxes_np[:, :4],
                        confidence=boxes_np[:, 4],
                    )
                    tracked = tracker.update_with_detections(detections)
                    _dbg_total_tracked += len(tracked)

                    for j in range(len(tracked)):
                        # Match tracked box back to original detection by closest distance
                        dists = np.sum((boxes_np[:, :4] - tracked.xyxy[j]) ** 2, axis=1)
                        orig_idx = np.argmin(dists)
                        pid = int(tracked.tracker_id[j])

                        if pid not in tracks:
                            tracks[pid] = {"frames": [], "poses3d": [], "poses2d": [], "confidences": []}
                        tracks[pid]["frames"].append(frame_idx)
                        tracks[pid]["poses3d"].append(poses3d_np[orig_idx])
                        tracks[pid]["poses2d"].append(poses2d_np[orig_idx])
                        tracks[pid]["confidences"].append(float(boxes_np[orig_idx, 4]))

                else:
                    _dbg_empty_frames += 1

                frame_idx += 1

            else:
                # Single person: keep largest bounding box
                if len(boxes_np) > 1:
                    areas = (boxes_np[:, 2] - boxes_np[:, 0]) * (boxes_np[:, 3] - boxes_np[:, 1])
                    best = np.argmax(areas)
                    poses3d_np = poses3d_np[best:best + 1]
                    poses2d_np = poses2d_np[best:best + 1]
                    boxes_np = boxes_np[best:best + 1]

                all_poses3d.append(poses3d_np)
                all_poses2d.append(poses2d_np)
                all_confidences.append(boxes_np[:, 4] if len(boxes_np) > 0 else np.array([]))

    if USE_POSEVIZ:
        viz.close()

    # ── Output ───────────────────────────────────────────────────────
    video_name = Path(video_filepath).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"{video_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.multi_person:
        print(f"\nDiagnostic tracking:")
        print(f"  Total frames: {frame_idx}")
        print(f"  Frames with detections: {frame_idx - _dbg_empty_frames}/{frame_idx}")
        print(f"  Total raw detections: {_dbg_total_detections}")
        print(f"  Total tracked detections: {_dbg_total_tracked}")
        print(f"  Avg detections/frame: {_dbg_total_detections / max(1, frame_idx - _dbg_empty_frames):.1f}")
        export_multi_person(tracks, frame_idx, joint_names, fps, output_dir, args)
    else:
        export_single_person(all_poses3d, all_poses2d, all_confidences, joint_names, fps, output_dir, args)


# ── Export functions ─────────────────────────────────────────────────

def export_single_person(all_poses3d, all_poses2d, all_confidences, joint_names, fps, output_dir, args):
    """Export single-person results (original behavior)."""
    all_poses3d = butterworth_filter_poses(all_poses3d, fps, cutoff_freq=6.0)
    all_poses3d = reorient_and_ground(all_poses3d, joint_names, subject_height=args.height)
    if args.stationary:
        all_poses3d = stabilize_jump(all_poses3d, list(joint_names), fps)

    save_to_trc(output_dir / "poses3d.trc", all_poses3d, joint_names, fps)
    save_to_json(output_dir / "results.json", all_poses3d, all_poses2d, all_confidences, joint_names, fps)

    print(f"\nTerminé ! Résultats sauvegardés dans : {output_dir}/")
    print(f"  - poses3d.trc (biomécanique/OpenSim)")
    print(f"  - results.json (poses3d, poses2d, confidences)")

    if args.ik:
        from kinematics import run_kinematics
        scaled_model, mot_file = run_kinematics(
            trc_file=output_dir / "poses3d.trc",
            output_dir=output_dir,
            subject_mass=args.mass,
            subject_height=args.height,
        )
        print(f"  - {scaled_model.name} (modèle OpenSim scalé)")
        print(f"  - {mot_file.name} (angles articulaires IK)")


def export_multi_person(tracks, total_frames, joint_names, fps, output_dir, args):
    """Export multi-person results: one subfolder per tracked person.

    Persons are sorted left-to-right based on their pelvis X position on their
    first detected frame, matching the order of --person_heights/--person_masses.
    """
    num_joints = len(joint_names)
    min_track_frames = max(5, int(fps * args.min_track_seconds))
    exported_tracks = []  # collect (person_count, poses3d_list, start_frame, end_frame) for combined TRC

    # Parse per-person height/mass lists (fall back to single --height/--mass)
    person_heights = _parse_float_list(args.person_heights) if args.person_heights else None
    person_masses = _parse_float_list(args.person_masses) if args.person_masses else None
    if person_heights:
        print(f"  Heights per person (left→right): {person_heights}")
    if person_masses:
        print(f"  Masses per person (left→right): {person_masses}")

    # Find pelvis joint index for sorting
    joint_list = list(joint_names)
    pelv_i = _joint_idx(joint_list, 'pelv', 'mhip')
    if pelv_i is None:
        pelv_i = 0  # fallback

    def track_first_x(track):
        """X position of pelvis on the track's first detected frame (for sorting)."""
        if not track["poses2d"]:
            return 0.0
        return float(track["poses2d"][0][pelv_i, 0])

    # Sort tracks left-to-right by X position on first detected frame
    sorted_tracks = sorted(tracks.items(), key=lambda kv: track_first_x(kv[1]))

    # Diagnostic
    print(f"\nTracking: {len(tracks)} track(s) trouvé(s) (triés gauche→droite)")
    for tid, t in sorted_tracks:
        x_first = track_first_x(t)
        print(f"  track {tid}: {len(t['frames'])} frames "
              f"(range {t['frames'][0]}-{t['frames'][-1]}, x={x_first:.0f}px)"
              f"{' [SKIPPED < ' + str(min_track_frames) + ' frames]' if len(t['frames']) < min_track_frames else ''}")

    person_count = 0
    summary = {"fps": fps, "total_frames": total_frames, "persons": []}

    for track_id, track in sorted_tracks:
        if len(track["frames"]) < min_track_frames:
            continue

        # Get per-person height/mass (fall back to --height/--mass defaults)
        this_height = person_heights[person_count] if person_heights and person_count < len(person_heights) else args.height
        this_mass = person_masses[person_count] if person_masses and person_count < len(person_masses) else args.mass

        # Build continuous pose array with interpolation
        poses3d_interp, start_frame, end_frame = interpolate_track(
            track, total_frames, num_joints)
        if poses3d_interp is None:
            continue

        # Wrap as list of (1, 87, 3) arrays for compatibility with existing functions
        poses3d_list = [poses3d_interp[i:i + 1] for i in range(len(poses3d_interp))]

        # Filter then reorient and ground-calibrate (per person, using this_height)
        poses3d_list = butterworth_filter_poses(poses3d_list, fps, cutoff_freq=6.0)
        poses3d_list = reorient_and_ground(poses3d_list, joint_names, subject_height=this_height)
        if args.stationary:
            poses3d_list = stabilize_jump(poses3d_list, list(joint_names), fps)

        # Build poses2d and confidences for JSON (sparse, only known frames)
        poses2d_sparse = {f: p for f, p in zip(track["frames"], track["poses2d"])}
        conf_sparse = {f: c for f, c in zip(track["frames"], track["confidences"])}

        # Build full-length 2d poses and confidence lists
        poses2d_list = []
        conf_list = []
        for f in range(start_frame, end_frame + 1):
            if f in poses2d_sparse:
                poses2d_list.append(poses2d_sparse[f][np.newaxis])
                conf_list.append(np.array([conf_sparse[f]]))
            else:
                poses2d_list.append(np.zeros((1, num_joints, 2)))
                conf_list.append(np.array([0.0]))

        # Export
        person_dir = output_dir / f"person_{person_count}"
        person_dir.mkdir(parents=True, exist_ok=True)

        save_to_trc(person_dir / "poses3d.trc", poses3d_list, joint_names, fps,
                     start_frame=start_frame)
        save_to_json(person_dir / "results.json", poses3d_list, poses2d_list, conf_list,
                     joint_names, fps, start_frame=start_frame)

        # Keep data for combined TRC
        if args.combined_trc:
            exported_tracks.append((person_count, poses3d_list, start_frame, end_frame))

        n_detected = len(track["frames"])
        n_total = end_frame - start_frame + 1
        n_interp = n_total - n_detected
        summary["persons"].append({
            "person_id": person_count,
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames_detected": n_detected,
            "frames_interpolated": n_interp,
            "height_m": this_height,
            "mass_kg": this_mass,
        })

        print(f"  person_{person_count}/ : frames {start_frame}-{end_frame} "
              f"({n_detected} detected, {n_interp} interpolated)")

        if args.ik:
            from kinematics import run_kinematics
            scaled_model, mot_file = run_kinematics(
                trc_file=person_dir / "poses3d.trc",
                output_dir=person_dir,
                subject_mass=this_mass,
                subject_height=this_height,
            )
            print(f"    - {scaled_model.name} + {mot_file.name} (mass={this_mass}kg, height={this_height}m)")

        person_count += 1

    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Combined TRC (all persons in one file, markers prefixed p<N>_)
    if args.combined_trc and exported_tracks:
        save_combined_trc(output_dir / "poses3d_combined.trc", exported_tracks,
                          joint_names, fps, total_frames)
        print(f"  poses3d_combined.trc : all {person_count} persons in one file")

    print(f"\nTerminé ! {person_count} personne(s) exportée(s) dans : {output_dir}/")


# ── Stationary stabilization ──────────────────────────────────────────

def stabilize_jump(all_poses3d, joint_list, fps):
    """Fix horizontal drift from monocular depth estimation.

    - Centers pelvis on X and Z each frame (removes horizontal drift)
    - Keeps Y (vertical) as-is to preserve jump height and vertical movement
    - Detects flight phases (feet off ground) and logs them if present
    """
    pelv_i = _joint_idx(joint_list, 'pelv', 'mhip')
    lhee_i = _joint_idx(joint_list, 'lhee')
    rhee_i = _joint_idx(joint_list, 'rhee')
    ltoe_i = _joint_idx(joint_list, 'ltoe')
    rtoe_i = _joint_idx(joint_list, 'rtoe')

    if pelv_i is None:
        return all_poses3d

    # Compute pelvis position per frame for centering
    pelvis_x = []
    pelvis_z = []
    for frame_poses in all_poses3d:
        if len(frame_poses) > 0:
            pelvis_x.append(frame_poses[0][pelv_i, 0])
            pelvis_z.append(frame_poses[0][pelv_i, 2])
        else:
            pelvis_x.append(0.0)
            pelvis_z.append(0.0)

    # Use median pelvis position as the fixed center (from standing frames)
    center_x = np.median(pelvis_x)
    center_z = np.median(pelvis_z)

    # Center all frames: shift X and Z so pelvis stays at median position
    for i, frame_poses in enumerate(all_poses3d):
        if len(frame_poses) > 0:
            dx = frame_poses[0][pelv_i, 0] - center_x
            dz = frame_poses[0][pelv_i, 2] - center_z
            frame_poses[0][:, 0] -= dx
            frame_poses[0][:, 2] -= dz

    # Detect flight phases (all foot markers above ground threshold)
    foot_indices = [i for i in [lhee_i, rhee_i, ltoe_i, rtoe_i] if i is not None]
    if foot_indices:
        ground_threshold = 30.0  # mm above ground = in the air
        flight_frames = []
        for i, frame_poses in enumerate(all_poses3d):
            if len(frame_poses) > 0:
                min_foot_y = min(frame_poses[0][fi, 1] for fi in foot_indices)
                if min_foot_y > ground_threshold:
                    flight_frames.append(i)

        if flight_frames:
            ranges = _frames_to_ranges(flight_frames)
            ranges_str = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)
            # Compute max jump height (max pelvis Y during flight - pelvis Y at takeoff)
            max_heights = []
            for a, b in ranges:
                takeoff_y = all_poses3d[max(0, a - 1)][0][pelv_i, 1] if a > 0 else 0
                flight_max_y = max(all_poses3d[f][0][pelv_i, 1] for f in range(a, b + 1)
                                   if len(all_poses3d[f]) > 0)
                max_heights.append(flight_max_y - takeoff_y)
            print(f"  Stationary mode: horizontal drift removed, pelvis centered")
            print(f"  Flight phases detected: {ranges_str}")
            for idx, ((a, b), h) in enumerate(zip(ranges, max_heights)):
                dur = (b - a + 1) / fps
                print(f"    Flight {idx + 1}: frames {a}-{b} ({dur:.2f}s), "
                      f"pelvis rise: {h:.0f}mm")
        else:
            print(f"  Stationary mode: horizontal drift removed, pelvis centered")
    else:
        print(f"  Stationary mode: horizontal drift removed, pelvis centered")

    return all_poses3d


# ── Butterworth low-pass filter ───────────────────────────────────────

def butterworth_filter_poses(all_poses3d, fps, cutoff_freq=6.0, order=4):
    """Apply a zero-phase Butterworth low-pass filter on 3D pose trajectories.

    Args:
        all_poses3d: list of (N_persons, num_joints, 3) arrays, one per frame.
        fps: video frame rate.
        cutoff_freq: cutoff frequency in Hz (default 6 Hz, standard in biomechanics).
        order: filter order (default 4, gives -80 dB/decade rolloff).

    Returns:
        Filtered all_poses3d (same structure).
    """
    num_frames = len(all_poses3d)
    # filtfilt needs signal length > 3*max(len(a), len(b)) = 3*(order+1) for order-N Butterworth
    min_frames_needed = 3 * (order + 1) + 2
    if num_frames < min_frames_needed:
        return all_poses3d

    # Only filter first person per frame (consistent with TRC export)
    num_joints = all_poses3d[0].shape[1] if len(all_poses3d[0]) > 0 else 0
    if num_joints == 0:
        return all_poses3d

    # Stack into (num_frames, num_joints, 3) for first person
    stacked = np.zeros((num_frames, num_joints, 3))
    valid = np.ones(num_frames, dtype=bool)
    for i, frame_poses in enumerate(all_poses3d):
        if len(frame_poses) > 0:
            stacked[i] = frame_poses[0]
        else:
            valid[i] = False

    # If too many missing frames, skip filtering
    if valid.sum() < min_frames_needed:
        return all_poses3d

    # Fill missing frames with linear interp before filtering (filtfilt needs continuous data)
    for j in range(num_joints):
        for c in range(3):
            col = stacked[:, j, c]
            if not valid.all():
                xp = np.where(valid)[0]
                fp_vals = col[valid]
                col[:] = np.interp(np.arange(num_frames), xp, fp_vals)

    # Design Butterworth filter
    nyquist = fps / 2.0
    if cutoff_freq >= nyquist:
        cutoff_freq = nyquist * 0.9  # safety clamp
    b, a = butter(order, cutoff_freq / nyquist, btype='low')

    # Apply zero-phase filter per joint per axis
    for j in range(num_joints):
        for c in range(3):
            stacked[:, j, c] = filtfilt(b, a, stacked[:, j, c])

    # Write back
    for i in range(num_frames):
        if len(all_poses3d[i]) > 0:
            all_poses3d[i][0] = stacked[i]

    return all_poses3d


# ── Interpolation ────────────────────────────────────────────────────

def interpolate_track(track, total_frames, num_joints):
    """Build a continuous pose array from sparse track data with linear interpolation.

    Returns (poses3d, start_frame, end_frame) or (None, None, None) if insufficient data.
    """
    frames = np.array(track["frames"])
    poses = np.array(track["poses3d"])  # (N, 87, 3)

    if len(frames) < 2:
        return None, None, None

    first, last = int(frames[0]), int(frames[-1])
    length = last - first + 1

    # Initialize with NaN
    result = np.full((length, num_joints, 3), np.nan)

    # Fill known frames
    for f, p in zip(frames, poses):
        result[f - first] = p

    # Linear interpolation per joint per coordinate (within known range)
    for j in range(num_joints):
        for c in range(3):
            col = result[:, j, c]
            known_mask = ~np.isnan(col)
            if known_mask.sum() < 2:
                continue
            xp = np.where(known_mask)[0]
            fp_vals = col[known_mask]
            col[:] = np.interp(np.arange(length), xp, fp_vals)

    # Any remaining NaN → 0
    result = np.nan_to_num(result, nan=0.0)

    return result, first, last


# ── Coordinate transform ─────────────────────────────────────────────

def reorient_and_ground(all_poses3d, joint_names, subject_height=None):
    """Reorient poses from camera frame to OpenSim Y-up convention,
    auto-straighten the vertical axis, optionally rescale, and calibrate feet to ground (Y=0).

    Pipeline:
      1. Axis remap: camera (X-right, Y-down, Z-forward) → OpenSim (X-anterior, Y-up, Z-right)
      2. Detect standing frames (knee angle > 160°)
      3. Vertical straightening: rotate so ankle→pelvis vector (from standing frames) aligns with Y
      4. Height rescaling: if subject_height provided, rescale based on measured vs real height
      5. Ground calibration: shift so feet touch Y=0
    """
    joint_list = list(joint_names)

    # Step 1: Full axis remap camera -> OpenSim
    for i in range(len(all_poses3d)):
        if len(all_poses3d[i]) > 0:
            x_cam = all_poses3d[i][:, :, 0].copy()
            y_cam = all_poses3d[i][:, :, 1].copy()
            z_cam = all_poses3d[i][:, :, 2].copy()
            all_poses3d[i][:, :, 0] = -z_cam   # anterior
            all_poses3d[i][:, :, 1] = -y_cam   # up
            all_poses3d[i][:, :, 2] = -x_cam   # person's right

    # Step 2: Fix lateral flips from monocular depth ambiguity
    all_poses3d = correct_lateral_flips(all_poses3d, joint_list)

    # Step 3: Detect standing frames
    standing_frames = detect_standing_frames(all_poses3d, joint_list)

    # Step 3: Auto-straighten vertical axis using standing frames only
    all_poses3d = straighten_vertical(all_poses3d, joint_list, standing_frames)

    # Step 4: Height rescaling (if subject_height provided)
    if subject_height is not None and subject_height > 0:
        all_poses3d = rescale_to_height(all_poses3d, joint_list, standing_frames, subject_height)

    # Step 5: Ground calibration (feet at Y=0)
    foot_joints = [j for j, name in enumerate(joint_names)
                   if 'toe' in name.lower() or 'heel' in name.lower() or 'ankle' in name.lower()]

    foot_y_values = []
    for frame_poses in all_poses3d:
        if len(frame_poses) > 0:
            foot_y_values.append(frame_poses[0][foot_joints, 1].min())

    if foot_y_values:
        ground_level = np.percentile(foot_y_values, 5)
        for i in range(len(all_poses3d)):
            if len(all_poses3d[i]) > 0:
                all_poses3d[i][:, :, 1] -= ground_level

    return all_poses3d


def _joint_idx(joint_list, *names):
    """Return the index of the first matching joint name, or None."""
    for name in names:
        if name in joint_list:
            return joint_list.index(name)
    return None


def _knee_angle(pose, hip_i, knee_i, ankle_i):
    """Compute the knee angle (hip-knee-ankle) in degrees."""
    v_thigh = pose[hip_i] - pose[knee_i]
    v_shank = pose[ankle_i] - pose[knee_i]
    n1 = np.linalg.norm(v_thigh)
    n2 = np.linalg.norm(v_shank)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cos_angle = np.clip(np.dot(v_thigh, v_shank) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def correct_lateral_flips(all_poses3d, joint_list):
    """Detect and fix frames where left/right and/or front/back are flipped
    due to monocular depth ambiguity.

    Lateral (Z):  checks if lshom.Z > rshom.Z (shoulders swapped side)
    Anterior (X): checks if clavicle.X < lscapula.X (front behind back)

    When flipped, mirrors the affected axis relative to the pelvis center.
    """
    lsho_i = _joint_idx(joint_list, 'lshom', 'lsho')
    rsho_i = _joint_idx(joint_list, 'rshom', 'rsho')
    pelv_i = _joint_idx(joint_list, 'pelv', 'mhip')
    clav_i = _joint_idx(joint_list, 'clavicle')
    scap_i = _joint_idx(joint_list, 'lscapula', 'lback')

    if any(idx is None for idx in [lsho_i, rsho_i, pelv_i]):
        return all_poses3d

    can_check_ap = clav_i is not None and scap_i is not None

    # First pass: determine majority orientation for each axis
    lat_normal = 0
    lat_flipped = 0
    ap_normal = 0
    ap_flipped = 0
    for frame_poses in all_poses3d:
        if len(frame_poses) > 0:
            pose = frame_poses[0]
            if pose[rsho_i, 2] > pose[lsho_i, 2]:
                lat_normal += 1
            else:
                lat_flipped += 1
            if can_check_ap:
                if pose[clav_i, 0] > pose[scap_i, 0]:
                    ap_normal += 1
                else:
                    ap_flipped += 1

    # Lateral correction
    if lat_flipped > lat_normal:
        check_lat_flip = lambda pose: pose[rsho_i, 2] > pose[lsho_i, 2]
    else:
        check_lat_flip = lambda pose: pose[lsho_i, 2] > pose[rsho_i, 2]

    # Anterior-posterior correction
    if can_check_ap:
        if ap_flipped > ap_normal:
            check_ap_flip = lambda pose: pose[clav_i, 0] > pose[scap_i, 0]
        else:
            check_ap_flip = lambda pose: pose[clav_i, 0] < pose[scap_i, 0]
    else:
        check_ap_flip = lambda pose: False

    # Second pass: fix flipped frames
    lat_corrected = 0
    ap_corrected = 0
    for i, frame_poses in enumerate(all_poses3d):
        if len(frame_poses) > 0:
            pose = frame_poses[0]
            if check_lat_flip(pose):
                pelvis_z = pose[pelv_i, 2]
                pose[:, 2] = 2 * pelvis_z - pose[:, 2]
                lat_corrected += 1
            if check_ap_flip(pose):
                pelvis_x = pose[pelv_i, 0]
                pose[:, 0] = 2 * pelvis_x - pose[:, 0]
                ap_corrected += 1

    if lat_corrected > 0 or ap_corrected > 0:
        parts = []
        if lat_corrected > 0:
            parts.append(f"lateral {lat_corrected} frames")
        if ap_corrected > 0:
            parts.append(f"anterior-posterior {ap_corrected} frames")
        print(f"  Flip correction: {', '.join(parts)} (out of {len(all_poses3d)})")

    return all_poses3d


def detect_standing_frames(all_poses3d, joint_list, angle_threshold=160.0):
    """Detect frames where the person is standing upright (knee angle > threshold).

    Returns a list of frame indices where both knees are nearly extended.
    """
    lhip_i = _joint_idx(joint_list, 'lhip', 'lasis')
    rhip_i = _joint_idx(joint_list, 'rhip', 'rasis')
    lkne_i = _joint_idx(joint_list, 'lkne', 'lknem')
    rkne_i = _joint_idx(joint_list, 'rkne', 'rknem')
    lank_i = _joint_idx(joint_list, 'lank', 'lankm')
    rank_i = _joint_idx(joint_list, 'rank', 'rankm')

    if any(idx is None for idx in [lhip_i, rhip_i, lkne_i, rkne_i, lank_i, rank_i]):
        return list(range(len(all_poses3d)))  # fallback: all frames

    standing = []
    for i, frame_poses in enumerate(all_poses3d):
        if len(frame_poses) == 0:
            continue
        pose = frame_poses[0]
        left_angle = _knee_angle(pose, lhip_i, lkne_i, lank_i)
        right_angle = _knee_angle(pose, rhip_i, rkne_i, rank_i)
        if left_angle > angle_threshold and right_angle > angle_threshold:
            standing.append(i)

    # Log standing frame ranges
    if standing:
        ranges = _frames_to_ranges(standing)
        ranges_str = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)
        print(f"  Standing frames detected: {len(standing)}/{len(all_poses3d)} "
              f"(ranges: {ranges_str})")
    else:
        print(f"  Warning: no standing frames detected (threshold={angle_threshold}°), "
              f"using all frames for calibration")
        standing = list(range(len(all_poses3d)))

    return standing


def _frames_to_ranges(frames):
    """Convert [0,1,2,5,6,7,10] to [(0,2),(5,7),(10,10)]."""
    ranges = []
    start = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f == prev + 1:
            prev = f
        else:
            ranges.append((start, prev))
            start = f
            prev = f
    ranges.append((start, prev))
    return ranges


def straighten_vertical(all_poses3d, joint_list, standing_frames):
    """Rotate all poses so that the ankle→pelvis vector (from standing frames)
    aligns with the Y axis."""
    lank_i = _joint_idx(joint_list, 'lank', 'lankm')
    rank_i = _joint_idx(joint_list, 'rank', 'rankm')
    pelv_i = _joint_idx(joint_list, 'pelv', 'mhip')

    if any(idx is None for idx in [lank_i, rank_i, pelv_i]):
        return all_poses3d

    # Collect ankle→pelvis vectors from standing frames only
    up_vectors = []
    for i in standing_frames:
        if i < len(all_poses3d) and len(all_poses3d[i]) > 0:
            pose = all_poses3d[i][0]
            mid_ankle = (pose[lank_i] + pose[rank_i]) / 2.0
            pelvis = pose[pelv_i]
            vec = pelvis - mid_ankle
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                up_vectors.append(vec / norm)

    if len(up_vectors) < 3:
        return all_poses3d

    # Median direction (robust to outliers)
    median_up = np.median(up_vectors, axis=0)
    median_up /= np.linalg.norm(median_up)

    target = np.array([0.0, 1.0, 0.0])
    R = rotation_align(median_up, target)

    angle = np.arccos(np.clip(np.dot(median_up, target), -1, 1))
    if np.degrees(angle) < 0.5:
        return all_poses3d

    print(f"  Auto-straighten: {np.degrees(angle):.1f}° correction applied "
          f"(based on {len(up_vectors)} standing frames)")

    for i in range(len(all_poses3d)):
        if len(all_poses3d[i]) > 0:
            for p in range(len(all_poses3d[i])):
                all_poses3d[i][p] = (R @ all_poses3d[i][p].T).T

    return all_poses3d


def rescale_to_height(all_poses3d, joint_list, standing_frames, subject_height):
    """Rescale all poses so that the measured height matches the real subject height.

    Measures head-to-heel distance during standing frames and computes a scale factor.
    subject_height in meters, poses in mm.
    """
    head_i = _joint_idx(joint_list, 'head')
    lhee_i = _joint_idx(joint_list, 'lhee')
    rhee_i = _joint_idx(joint_list, 'rhee')

    if any(idx is None for idx in [head_i, lhee_i, rhee_i]):
        return all_poses3d

    # Measure head-to-ground height in standing frames
    heights = []
    for i in standing_frames:
        if i < len(all_poses3d) and len(all_poses3d[i]) > 0:
            pose = all_poses3d[i][0]
            head_y = pose[head_i, 1]
            heel_y = min(pose[lhee_i, 1], pose[rhee_i, 1])
            h = head_y - heel_y
            if h > 100:  # at least 100mm to be valid
                heights.append(h)

    if len(heights) < 3:
        print(f"  Height rescaling: not enough standing frames, skipped")
        return all_poses3d

    measured_height_mm = np.median(heights)
    target_height_mm = subject_height * 1000.0  # meters → mm
    scale_factor = target_height_mm / measured_height_mm

    print(f"  Height rescaling: measured {measured_height_mm:.0f}mm → "
          f"target {target_height_mm:.0f}mm (scale factor: {scale_factor:.3f})")

    if abs(scale_factor - 1.0) < 0.01:
        return all_poses3d  # negligible, skip

    for i in range(len(all_poses3d)):
        if len(all_poses3d[i]) > 0:
            all_poses3d[i] *= scale_factor

    return all_poses3d


def rotation_align(v_from, v_to):
    """Compute the 3x3 rotation matrix that aligns vector v_from to v_to.
    Uses Rodrigues' rotation formula."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)

    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return np.eye(3)  # already aligned
        else:
            # 180° rotation — pick an arbitrary perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v_from, perp)
            axis /= np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            return -np.eye(3) + 2 * np.outer(axis, axis)

    K = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]])

    R = np.eye(3) + K + K @ K * (1.0 / (1.0 + dot))
    return R


# ── File I/O ──────────────────────────────────────────────────────────

def get_video(source, temppath='/tmp/video.mp4'):
    if not source.startswith('http'):
        return source
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath


def save_to_json(filepath, all_poses3d, all_poses2d, all_confidences, joint_names, fps,
                 start_frame=0):
    """Save all data to a single readable JSON file."""
    data = {
        "fps": fps,
        "joint_names": joint_names.tolist(),
        "start_frame": start_frame,
        "frames": []
    }
    for i, (p3d, p2d, conf) in enumerate(zip(all_poses3d, all_poses2d, all_confidences)):
        frame_data = {
            "frame": start_frame + i + 1,
            "time": round((start_frame + i) / fps, 5),
            "num_persons": len(p3d),
            "persons": []
        }
        for j in range(len(p3d)):
            frame_data["persons"].append({
                "confidence": round(float(conf[j]), 4) if j < len(conf) else None,
                "poses3d": p3d[j].tolist(),
                "poses2d": p2d[j].tolist(),
            })
        data["frames"].append(frame_data)

    with open(filepath, 'w') as f:
        json.dump(data, f)


def save_to_trc(filepath, poses3d, joint_names, fps=30.0, start_frame=0):
    """Save poses to TRC format (Track Row Column) for biomechanics (e.g. OpenSim)."""
    num_frames = len(poses3d)
    num_markers = len(joint_names)

    with open(filepath, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tmm\t{fps}\t{start_frame + 1}\t{num_frames}\n")

        # Marker names (line 4)
        f.write("Frame#\tTime\t")
        for name in joint_names:
            f.write(f"{name}\t\t\t")
        f.write("\n")

        # Axes X1, Y1, Z1, X2, Y2, Z2... (line 5)
        f.write("\t\t")
        for i in range(num_markers):
            n = i + 1
            f.write(f"X{n}\tY{n}\tZ{n}\t")
        f.write("\n\n")

        # Data rows
        for i, frame_poses in enumerate(poses3d):
            t = (start_frame + i) / fps
            f.write(f"{start_frame + i + 1}\t{t:.5f}\t")

            pose = frame_poses[0] if len(frame_poses) > 0 else np.zeros((num_markers, 3))

            for j in range(num_markers):
                x, y, z = pose[j]
                f.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t")
            f.write("\n")


def save_combined_trc(filepath, exported_tracks, joint_names, fps, total_frames):
    """Save a single TRC file combining all tracked persons.

    Markers are prefixed with p<N>_ (e.g. p0_backneck, p1_backneck, ...).
    For frames where a person isn't present, their markers are set to 0.
    """
    num_persons = len(exported_tracks)
    num_joints = len(joint_names)
    num_markers = num_persons * num_joints

    # Build full-length data: shape (total_frames, num_markers, 3)
    data = np.zeros((total_frames, num_markers, 3))
    marker_names = []

    for pidx, (pcount, poses3d_list, start_f, end_f) in enumerate(exported_tracks):
        for j, name in enumerate(joint_names):
            marker_names.append(f"p{pcount}_{name}")
        for fi, frame_poses in enumerate(poses3d_list):
            abs_f = start_f + fi
            if abs_f < total_frames and len(frame_poses) > 0:
                data[abs_f, pidx * num_joints:(pidx + 1) * num_joints] = frame_poses[0]

    with open(filepath, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{total_frames}\t{num_markers}\tmm\t{fps}\t1\t{total_frames}\n")

        f.write("Frame#\tTime\t")
        for name in marker_names:
            f.write(f"{name}\t\t\t")
        f.write("\n")

        f.write("\t\t")
        for i in range(num_markers):
            n = i + 1
            f.write(f"X{n}\tY{n}\tZ{n}\t")
        f.write("\n\n")

        for i in range(total_frames):
            t = i / fps
            f.write(f"{i + 1}\t{t:.5f}\t")
            for j in range(num_markers):
                x, y, z = data[i, j]
                f.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t")
            f.write("\n")


if __name__ == '__main__':
    main()
