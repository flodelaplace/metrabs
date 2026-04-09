"""
OpenSim Scaling + Inverse Kinematics for MeTRAbs bml_movi_87 skeleton.

Inspired by Pose2Sim kinematics.py (David Pagnon, Ivan Sun).
Simplified standalone version: takes a TRC file, scales the model, runs IK.

Requires: opensim (conda install -c opensim-org opensim), lxml
"""

import os
import logging
from pathlib import Path
from lxml import etree

import opensim
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── paths (all resolved to absolute) ─────────────────────────────────
SETUP_DIR = Path(__file__).resolve().parent.parent / 'OpenSim_Setup'
MODEL_FILE = SETUP_DIR / 'Model_Pose2Sim_muscles_flex.osim'
MARKERS_FILE = SETUP_DIR / 'Markers_bml_movi_87.xml'
SCALING_SETUP_FILE = SETUP_DIR / 'Scaling_Setup_bml_movi_87.xml'
IK_SETUP_FILE = SETUP_DIR / 'IK_Setup_bml_movi_87.xml'
GEOMETRY_DIR = SETUP_DIR / 'Geometry'


def read_trc_time_range(trc_path):
    """Read a TRC file and return (start_time, end_time, num_frames)."""
    times = []
    data_started = False
    with open(trc_path, 'r') as f:
        for line in f:
            if data_started:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        times.append(float(parts[1]))
                    except ValueError:
                        continue
            # Data lines start with an integer frame number
            if not data_started:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    try:
                        int(parts[0])
                        times.append(float(parts[1]))
                        data_started = True
                    except ValueError:
                        pass

    if not times:
        raise ValueError(f"Could not read time data from TRC file: {trc_path}")

    return times[0], times[-1], len(times)


def perform_scaling(trc_file, output_dir, subject_mass=69, subject_height=1.75):
    """
    Scale the OpenSim model based on marker distances in the TRC file.

    Returns the path to the scaled .osim model.
    """
    # Resolve all paths to absolute to avoid OpenSim relative path issues
    trc_file = Path(trc_file).resolve()
    output_dir = Path(output_dir).resolve()

    # Register geometry search path
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(GEOMETRY_DIR))

    # Load unscaled model and add marker set programmatically
    logger.info(f"Loading model: {MODEL_FILE.name}")
    model = opensim.Model(str(MODEL_FILE))
    marker_set = opensim.MarkerSet(str(MARKERS_FILE))
    model.set_MarkerSet(marker_set)
    model.initSystem()

    # Save model with markers embedded (marker_set_file stays Unassigned in XML)
    scaled_model_path = output_dir / (trc_file.stem + '_scaled.osim')
    model.printToXML(str(scaled_model_path))
    logger.info(f"Unscaled model with markers saved: {scaled_model_path.name}")

    # Read time range from TRC
    start_time, end_time, _ = read_trc_time_range(trc_file)

    # Parse and update scaling setup XML
    scaling_tree = etree.parse(str(SCALING_SETUP_FILE))
    scaling_root = scaling_tree.getroot()

    # Update model path (markers already embedded, no need for marker_set_file)
    scaling_root[0].find('GenericModelMaker').find('model_file').text = str(scaled_model_path)

    # Update mass and height
    scaling_root[0].find('mass').text = str(subject_mass)
    scaling_root[0].find('height').text = str(subject_height * 1000)  # mm

    # Update marker file (TRC) and time range
    for mk_f in scaling_root[0].findall('.//marker_file'):
        mk_f.text = str(trc_file)
    scaling_root[0].find('.//ModelScaler').find('time_range').text = f'{start_time} {end_time}'
    scaling_root[0].find('.//ModelScaler').find('output_model_file').text = str(scaled_model_path)

    # Write temp scaling setup (absolute path critical for OpenSim path resolution)
    scaling_setup_temp = output_dir / (trc_file.stem + '_scaling_setup.xml')
    etree.indent(scaling_tree, space='\t', level=0)
    scaling_tree.write(str(scaling_setup_temp), pretty_print=True, xml_declaration=True, encoding='utf-8')

    # Run scaling
    logger.info("Running OpenSim ScaleTool...")
    opensim.ScaleTool(str(scaling_setup_temp)).run()
    logger.info(f"Scaled model saved: {scaled_model_path.name}")

    # Clean up temp file
    scaling_setup_temp.unlink(missing_ok=True)

    return scaled_model_path


def perform_ik(trc_file, output_dir):
    """
    Run OpenSim Inverse Kinematics on the scaled model.

    Returns the path to the .mot output file.
    """
    trc_file = Path(trc_file).resolve()
    output_dir = Path(output_dir).resolve()

    scaled_model_path = output_dir / (trc_file.stem + '_scaled.osim')
    if not scaled_model_path.exists():
        raise FileNotFoundError(f"Scaled model not found: {scaled_model_path}")

    # Output .mot path
    mot_file = output_dir / (trc_file.stem + '_ik.mot')

    # Read time range from TRC
    start_time, end_time, _ = read_trc_time_range(trc_file)

    # Parse and update IK setup XML
    ik_tree = etree.parse(str(IK_SETUP_FILE))
    ik_root = ik_tree.getroot()

    ik_root.find('.//model_file').text = str(scaled_model_path)
    ik_root.find('.//time_range').text = f'{start_time} {end_time}'
    ik_root.find('.//output_motion_file').text = str(mot_file)
    ik_root.find('.//marker_file').text = str(trc_file)

    # Write temp IK setup (absolute path)
    ik_setup_temp = output_dir / (trc_file.stem + '_ik_setup.xml')
    ik_tree.write(str(ik_setup_temp), pretty_print=True, xml_declaration=True, encoding='utf-8')

    # Run IK
    logger.info("Running OpenSim Inverse Kinematics...")
    opensim.InverseKinematicsTool(str(ik_setup_temp)).run()
    logger.info(f"IK results saved: {mot_file.name}")

    # Clean up temp file
    ik_setup_temp.unlink(missing_ok=True)

    return mot_file


def run_kinematics(trc_file, output_dir, subject_mass=69, subject_height=1.75):
    """
    Full pipeline: scaling + IK.

    Returns:
        (scaled_model_path, mot_file_path)
    """
    logger.info("=" * 50)
    logger.info("OpenSim Kinematics Pipeline")
    logger.info("=" * 50)

    # Step 1: Scaling
    scaled_model = perform_scaling(trc_file, output_dir,
                                   subject_mass=subject_mass,
                                   subject_height=subject_height)

    # Step 2: Inverse Kinematics
    mot_file = perform_ik(trc_file, output_dir)

    logger.info("=" * 50)
    logger.info("Kinematics pipeline complete!")
    logger.info(f"  Scaled model: {scaled_model.name}")
    logger.info(f"  Joint angles: {mot_file.name}")
    logger.info("=" * 50)

    return scaled_model, mot_file
