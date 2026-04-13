# MeTRAbs Fork - Video-to-Biomechanics Pipeline

Fork of [isarandi/metrabs](https://github.com/isarandi/metrabs) for extracting 3D poses from monocular video and producing biomechanical outputs (TRC, OpenSim IK).

> **MeTRAbs** (Metric-Scale Truncation-Robust Heatmaps) estimates absolute 3D human poses from single images using the `bml_movi_87` skeleton (87 joints).

---

## Features

| Feature | Description |
|---|---|
| **3D pose estimation** | MeTRAbs Large model, `bml_movi_87` skeleton (87 joints) |
| **TRC export** | OpenSim-compatible marker file |
| **JSON export** | Full data: poses3d, poses2d, detection confidence |
| **OpenSim IK** | Automatic scaling + Inverse Kinematics (`--ik`) |
| **Multi-person** | ByteTrack tracking with per-person output (`--multi_person`), sorted left-to-right |
| **Per-person anthropometry** | Specify height/mass per person with `--person_heights`/`--person_masses` |
| **Combined TRC** | Single TRC with all persons (markers prefixed `p0_`, `p1_`, ...) for easy visualization (`--combined_trc`) |
| **Butterworth filter** | 6 Hz low-pass, zero-phase (biomechanics standard) |
| **Auto-straighten** | Vertical calibration from standing frames |
| **Height rescaling** | Scale TRC to real subject height (`--height`) |
| **Flip correction** | Fixes lateral and anterior-posterior depth ambiguity |
| **Stationary mode** | Centers pelvis horizontally, detects flight phases (`--stationary`) |
| **Person selection** | Largest bounding box (single-person) or ByteTrack (multi-person) |

---

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04 / WSL2)
- NVIDIA GPU with recent drivers (tested with RTX 3500 Ada)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 1. Clone and checkout

```bash
git clone https://github.com/flodelaplace/metrabs.git
cd metrabs
git checkout dev
```

### 2. Create the conda environment

```bash
conda env create --name=metrabs --file=environment.yml
conda activate metrabs
```

### 3. Configure GPU (CUDA)

TensorFlow installed via pip needs the CUDA libraries path set manually:

```bash
# Set up automatic LD_LIBRARY_PATH on activation
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Reactivate
conda deactivate
conda activate metrabs
```

Verify GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 4. Install OpenSim (optional, for `--ik`)

```bash
conda install -c opensim-org opensim
```

Verify:

```bash
python -c "import opensim; print(opensim.GetVersion())"
```

### Tested versions

| Component | Version |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.12.0 |
| CUDA | 11.8 |
| cuDNN | 8.9 |
| OpenSim | 4.5+ |
| supervision | 0.27+ |
| NVIDIA Driver | 581+ |

---

## Usage

### Basic (single person, TRC + JSON output)

```bash
conda activate metrabs
python demos/demo_video.py path/to/video.mp4
```

### With OpenSim Inverse Kinematics

```bash
python demos/demo_video.py video.mp4 --ik --height 1.80 --mass 75
```

### Multi-person tracking

```bash
python demos/demo_video.py video.mp4 --multi_person
```

### Stationary movement (squat, CMJ, gait on treadmill...)

```bash
python demos/demo_video.py video.mp4 --stationary --ik --height 1.80 --mass 75
```

### All options

```
usage: demo_video.py [-h] [--ik] [--multi_person] [--stationary]
                     [--combined_trc]
                     [--min_track_seconds MIN_TRACK_SECONDS]
                     [--mass MASS] [--height HEIGHT]
                     [--person_masses PERSON_MASSES]
                     [--person_heights PERSON_HEIGHTS]
                     video

positional arguments:
  video                  Path to video file or URL

optional arguments:
  -h, --help             show this help message and exit
  --ik                   Run OpenSim scaling + Inverse Kinematics
  --multi_person         Track and export all persons separately (ByteTrack)
  --stationary           Fix horizontal drift (centers pelvis, detects flight phases)
  --combined_trc         Multi-person: also output a single combined TRC
                         (markers prefixed p0_, p1_, ...)
  --min_track_seconds N  Minimum track duration in seconds for multi-person
                         (default: 2.0; raise for noisy long videos)
  --mass MASS            Subject mass in kg, single-person (default: 69)
  --height HEIGHT        Subject height in meters, single-person (default: 1.75)
  --person_masses LIST   Multi-person: comma-separated masses in kg, sorted
                         left-to-right by pelvis X on first detected frame
                         (e.g. "55,60,58,70,80,75")
  --person_heights LIST  Multi-person: comma-separated heights in meters,
                         sorted left-to-right (e.g. "1.62,1.64,1.61,1.70,1.78,1.75")
```

### Per-person heights and masses (multi-person)

In `--multi_person` mode, persons are sorted **left-to-right** by their pelvis X position on their first detected frame. The lists `--person_heights` and `--person_masses` follow this order:

```bash
# 6 people, heights and masses left-to-right
python demos/demo_video.py video.mp4 \
    --multi_person --ik --combined_trc \
    --person_heights 1.62,1.64,1.61,1.70,1.78,1.75 \
    --person_masses 55,60,58,70,80,75
```

If the list is shorter than the number of detected persons, extras fall back to `--height` / `--mass`. The order is logged at runtime and saved in `summary.json` per person.

### Examples

```bash
# Simple extraction
python demos/demo_video.py img/WhatsAppVideoSquat.mp4

# Full biomechanics pipeline (single person)
python demos/demo_video.py img/video.mp4 --ik --height 1.85 --mass 80 --stationary

# Multi-person with IK per person (same height/mass for all)
python demos/demo_video.py img/video.mp4 --multi_person --ik --height 1.75 --mass 70

# Multi-person with per-person anthropometry + combined TRC for visualization
python demos/demo_video.py img/video.mp4 --multi_person --ik --combined_trc \
    --person_heights 1.62,1.64,1.61,1.70,1.78,1.75 \
    --person_masses 55,60,58,70,80,75

# Long noisy video: keep only tracks longer than 5 seconds
python demos/demo_video.py img/video.mp4 --multi_person --min_track_seconds 5
```

---

## Output

### Single person

```
output/<video_name>_<YYYYMMDD_HHMMSS>/
    poses3d.trc              # TRC markers for OpenSim (mm, Y-up)
    results.json             # Full data (poses3d, poses2d, confidence)
    poses3d_scaled.osim      # Scaled OpenSim model       (if --ik)
    poses3d_ik.mot           # Joint angles from IK        (if --ik)
```

### Multi-person (`--multi_person`)

```
output/<video_name>_<YYYYMMDD_HHMMSS>/
    summary.json                 # Track info (frames, durations, height, mass per person)
    poses3d_combined.trc         # All persons in one TRC (if --combined_trc)
    person_0/                    # Leftmost person on first frame
        poses3d.trc
        results.json
        poses3d_scaled.osim      (if --ik)
        poses3d_ik.mot           (if --ik)
    person_1/                    # Next person to the right
        ...
```

**Combined TRC** (`--combined_trc`): a single `poses3d_combined.trc` aggregating all persons. Each marker is prefixed with the person index (e.g. `p0_backneck`, `p1_backneck`, ...). Total markers = N_persons × 87. Useful for visualizing all persons simultaneously in OpenSim GUI or other TRC viewers. Note: this combined TRC is **not** suitable for OpenSim IK (use the per-person `poses3d.trc` files for IK).

### JSON structure

```json
{
  "fps": 30.0,
  "joint_names": ["backneck", "upperback", "..."],
  "start_frame": 0,
  "frames": [
    {
      "frame": 1,
      "time": 0.0,
      "num_persons": 1,
      "persons": [
        {
          "confidence": 0.9532,
          "poses3d": [[x, y, z], ...],
          "poses2d": [[x, y], ...]
        }
      ]
    }
  ]
}
```

> **Note:** Confidence is per detection (person-level), not per joint. MeTRAbs does not provide per-joint confidence scores.

---

## Processing Pipeline

The pipeline applies these steps in order:

```
Video frames
    |
    v
[MeTRAbs inference] -----> raw 3D poses (camera frame, mm)
    |
    v
[Person selection]  -----> largest bbox (single) or ByteTrack (multi)
    |
    v
[Butterworth filter] ----> 6 Hz low-pass, 4th order, zero-phase
    |
    v
[Axis remap] ------------> camera (X-right, Y-down, Z-forward)
    |                       to OpenSim (X-anterior, Y-up, Z-right)
    v
[Flip correction] -------> fixes lateral + anterior-posterior depth ambiguity
    |
    v
[Standing detection] ----> knee angle > 160 on both legs
    |
    v
[Vertical straighten] ---> rotates so ankle-pelvis aligns with Y (from standing frames)
    |
    v
[Height rescaling] ------> scales head-to-heel to match --height (from standing frames)
    |
    v
[Ground calibration] ----> shifts feet to Y=0 (5th percentile)
    |
    v
[Stationary centering] --> centers pelvis on X,Z (if --stationary)
    |
    v
[TRC + JSON export]
    |
    v
[OpenSim Scaling + IK] --> .osim + .mot (if --ik)
```

### Coordinate system

| Axis | OpenSim convention | Camera convention |
|---|---|---|
| X | Anterior (forward) | Image right |
| Y | Superior (up) | Image down |
| Z | Person's right (lateral) | Depth (forward into scene) |

The axis remap assumes the **subject faces the camera**. Mapping: `X_osim = -Z_cam`, `Y_osim = -Y_cam`, `Z_osim = -X_cam`.

---

## OpenSim Integration

### Marker set

The `bml_movi_87` marker set (`OpenSim_Setup/Markers_bml_movi_87.xml`) maps all 87 joints to OpenSim body segments using a hybrid approach:

- **Anatomical markers** (55): positions from the SKEL model (lab-grade surface landmarks)
- **Joint centers** (19): positions from HALPE_26 and LSTM models (joint center estimates)
- **Custom** (13): estimated positions for markers without direct equivalents

### Model

Uses `Model_Pose2Sim_muscles_flex.osim` (Pose2Sim model with muscles and constraints, 62 DOF, 318 muscles).

### Setup files

| File | Purpose |
|---|---|
| `Markers_bml_movi_87.xml` | 87 markers with body attachments and positions |
| `Scaling_Setup_bml_movi_87.xml` | 11 measurement pairs for segment scaling |
| `IK_Setup_bml_movi_87.xml` | 87 IK tasks with weights (pelvis 25, knee 30, heel 60, etc.) |
| `Model_Pose2Sim_muscles_flex.osim` | Unscaled musculoskeletal model |
| `Geometry/` | 215 mesh files for model visualization |

### IK weights strategy

| Body region | Weight | Rationale |
|---|---|---|
| Pelvis (ASIS, PSIS, hip centers) | 25 | Anchors the model |
| Knees, ankles | 30 | Critical for gait analysis |
| Heels | 60 | Ground contact reference |
| Shoulders, elbows, wrists | 5 | Upper body articulation |
| Thigh, shin, forearm | 4 | Segment tracking |
| Torso (C7, scapula, back) | 2-5 | Trunk orientation |
| Hands (fingers, thumb) | 1 | Low priority |
| Head | 0.5 | Low reliability from video |

---

## Multi-Person Tracking

When `--multi_person` is used:

1. **Detection**: MeTRAbs detects all persons per frame with bounding boxes
2. **Tracking**: [ByteTrack](https://github.com/ifzhang/ByteTrack) (via [supervision](https://github.com/roboflow/supervision)) assigns stable IDs across frames
3. **Filtering**: tracks shorter than 0.5 seconds are discarded
4. **Interpolation**: missing frames within a track are linearly interpolated per joint
5. **Export**: each person gets their own subfolder with independent post-processing

### Tracker parameters

| Parameter | Value | Description |
|---|---|---|
| `track_activation_threshold` | 0.1 | Min confidence to start a track |
| `lost_track_buffer` | 60 | Keep lost tracks for 2 seconds |
| `minimum_matching_threshold` | 0.3 | IoU threshold for matching |
| `minimum_consecutive_frames` | 1 | Track from first detection |

---

## Known Limitations

| Limitation | Description | Workaround |
|---|---|---|
| **Monocular depth** | Absolute position (especially depth) is estimated from apparent size. Can drift during fast movements. | Use `--stationary` for fixed-camera setups |
| **Jump height** | Vertical displacement during flight is exaggerated (depth/height confusion) | Joint angles (IK) are still valid |
| **Depth ambiguity** | Occasional front/back or left/right flip in unusual poses | Auto-corrected by flip detection |
| **Facing direction** | Axis remap assumes subject faces camera | Rotate video or adjust code for other orientations |
| **Single camera** | No true 3D triangulation | Use Pose2Sim for multi-camera setups |

---

## Performance

Benchmarked on RTX 3500 Ada (12 GB VRAM):

| Metric | Value |
|---|---|
| Inference speed (GPU) | ~0.8 s/batch (8 frames) |
| Inference speed (CPU) | ~11 s/batch |
| Butterworth filter | < 0.1s for 1000 frames |
| OpenSim Scaling | ~2s |
| OpenSim IK | ~5s for 300 frames |

---

## Project Structure

```
metrabs/
    demos/
        demo_video.py          # Main pipeline script
        kinematics.py          # OpenSim scaling + IK module
    OpenSim_Setup/
        Markers_bml_movi_87.xml
        Scaling_Setup_bml_movi_87.xml
        IK_Setup_bml_movi_87.xml
        Model_Pose2Sim_muscles_flex.osim
        Geometry/              # 215 mesh files
    environment.yml            # Conda environment definition
    FORK_README.md             # This file
    README.md                  # Original MeTRAbs README
```

---

## Credits

This fork is based on [MeTRAbs](https://github.com/isarandi/metrabs) by [Istvan Sarandi](https://isarandi.github.io) et al.:

- [MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://arxiv.org/abs/2007.07227) (T-BIOM 2021)
- [Learning 3D Human Pose Estimation from Dozens of Datasets](https://arxiv.org/abs/2212.14474) (WACV 2023)

OpenSim model and marker set design inspired by [Pose2Sim](https://github.com/perfanalytics/pose2sim) (David Pagnon et al.).

Multi-person tracking powered by [supervision](https://github.com/roboflow/supervision) (Roboflow).
