# MeTRAbs Fork - Video Pipeline for Biomechanical Analysis

Fork of [isarandi/metrabs](https://github.com/isarandi/metrabs) adapted for 3D pose extraction from videos, with export to biomechanical formats (TRC for OpenSim) and JSON.

## Changes from the Original Repo

### `demos/demo_video.py`

| Aspect | Original | This fork |
|---|---|---|
| Video reading | `tensorflow_io` (often hard to install) | `imageio` + `ffmpeg` (more portable) |
| Skeleton | `smpl_24` (24 joints) | `bml_movi_87` (87 joints, more detailed) |
| Outputs | PoseViz display only | Export **TRC** + **JSON** (poses3d, poses2d, confidence) |
| Output organization | No file saving | `output/<video_name>_<YYYYMMDD_HHMMSS>/` folder (no overwriting) |
| Progress bar | No | Yes (`tqdm`) |
| PoseViz | Always active | Toggle `USE_POSEVIZ` (disabled by default) |

### `environment.yml`

- Python 3.9 → **3.10**
- Removed `smpl` dependency (not needed for inference with `bml_movi_87`)

### Output Files

Each run creates a unique folder in `output/`:

```
output/
└── MyVideo_20260408_143000/
    ├── poses3d.trc          # TRC format for OpenSim
    └── results.json         # All data (poses3d, poses2d, confidence)
```

**JSON structure:**
```json
{
  "fps": 30.0,
  "joint_names": ["joint1", "joint2", "..."],
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

> **Note:** Confidence is at the detection level (per person), not per individual keypoint. MeTRAbs does not provide per-joint confidence scores.

---

## Installation

### Prerequisites

- Linux (tested on Ubuntu 22.04 / WSL2)
- NVIDIA GPU with recent drivers (tested with RTX 3500 Ada)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### 1. Clone the repo

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

### 3. Configure GPU support (important)

TensorFlow installed via pip does not automatically find CUDA libraries from conda. You need to add the path manually.

**Install CUDA packages in conda:**
```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9
```

**Set up automatic `LD_LIBRARY_PATH` configuration:**
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Then reactivate the environment:
```bash
conda deactivate
conda activate metrabs
```

**Verify GPU is detected:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should print `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.

### 4. Tested versions

| Component | Version |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.12.0 |
| CUDA | 11.8 |
| cuDNN | 8.9 |
| NVIDIA Driver | 581+ |

---

## Usage

```bash
conda activate metrabs
python demos/demo_video.py path/to/video.mp4
```

**Example:**
```bash
python demos/demo_video.py img/WhatsAppVideoSquat.mp4
```

**Performance benchmark (RTX 3500 Ada, 12 GB VRAM):**
- ~0.8 s/batch (batches of 8 frames) on GPU
- ~11 s/batch on CPU (without GPU configured)

---

## Credits

This fork is based on the work of [István Sárándi](https://isarandi.github.io) et al.:

- [MeTRAbs: Metric-Scale Truncation-Robust Heatmaps for Absolute 3D Human Pose Estimation](https://arxiv.org/abs/2007.07227) (T-BIOM 2021)
- [Learning 3D Human Pose Estimation from Dozens of Datasets](https://arxiv.org/abs/2212.14474) (WACV 2023)

See the [original README](https://github.com/isarandi/metrabs) for BibTeX citations.
