import sys
import os
import json
import urllib.request
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as tfhub
import imageio
import numpy as np
from tqdm import tqdm

import cameralib
import poseviz


def main():
    USE_POSEVIZ = False  # Mets à True si tu veux réactiver l'affichage visuel

    # On utilise le modèle Large (très rapide maintenant que le GPU est activé) :
    model = tfhub.load('https://bit.ly/metrabs_l')
    skeleton = 'bml_movi_87'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    video_filepath = get_video(sys.argv[1])  # You can also specify the filepath directly here.

    reader = imageio.get_reader(video_filepath, 'ffmpeg')
    fps = reader.get_meta_data().get('fps', 30.0)
    imshape = reader.get_data(0).shape[:2]
    
    try:
        total_batches = int(np.ceil(reader.count_frames() / 8))
    except:
        total_batches = None  # Sécurité si la vidéo n'expose pas son nombre total de frames

    camera = cameralib.Camera.from_fov(fov_degrees=55, imshape=imshape)

    def frame_generator():
        for frame in imageio.get_reader(video_filepath, 'ffmpeg'):
            yield frame

    frame_batches = tf.data.Dataset.from_generator(
        frame_generator,
        output_signature=tf.TensorSpec(shape=(imshape[0], imshape[1], 3), dtype=tf.uint8)
    ).batch(8).prefetch(1)

    all_poses3d = []
    all_poses2d = []
    all_confidences = []

    if USE_POSEVIZ:
        viz = poseviz.PoseViz(joint_names, joint_edges)

    for i, frame_batch in enumerate(tqdm(frame_batches, total=total_batches, desc="Inférence 3D en cours", unit="batch")):
        pred = model.detect_poses_batched(
            frame_batch, intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
            skeleton=skeleton)
        
        # On extrait les poses image par image dans le batch
        for frame, boxes, poses3d, poses2d in zip(
                frame_batch, pred['boxes'], pred['poses3d'], pred['poses2d']):
            all_poses3d.append(poses3d.numpy())
            all_poses2d.append(poses2d.numpy())
            # confidence = 5e colonne des boxes (par personne détectée)
            all_confidences.append(boxes.numpy()[:, 4] if len(boxes) > 0 else np.array([]))

            if USE_POSEVIZ:
                viz.update(frame=frame, boxes=boxes, poses=poses3d, camera=camera)

    if USE_POSEVIZ:
        viz.close()

    # Post-processing: reorient to Y-up and ground-calibrate feet
    all_poses3d = reorient_and_ground(all_poses3d, joint_names)

    # Créer le dossier de sortie : output/<nom_video>_<datetime>/
    video_name = Path(video_filepath).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / f"{video_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde
    save_to_trc(output_dir / "poses3d.trc", all_poses3d, joint_names, fps)
    save_to_json(output_dir / "results.json", all_poses3d, all_poses2d, all_confidences, joint_names, fps)

    print(f"\nTerminé ! Résultats sauvegardés dans : {output_dir}/")
    print(f"  - poses3d.trc (biomécanique/OpenSim)")
    print(f"  - results.json (poses3d, poses2d, confidences)")


def get_video(source, temppath='/tmp/video.mp4'):
    if not source.startswith('http'):
        return source

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath

def reorient_and_ground(all_poses3d, joint_names):
    """Reorient poses from camera frame (Y-down) to OpenSim convention (Y-up),
    and offset vertically so that feet touch the ground (Y=0)."""
    # Step 1: Flip Y axis (camera Y-down -> Y-up)
    for i in range(len(all_poses3d)):
        if len(all_poses3d[i]) > 0:
            all_poses3d[i][:, :, 1] *= -1

    # Step 2: Find foot joint indices
    foot_joints = [j for j, name in enumerate(joint_names)
                   if 'toe' in name.lower() or 'heel' in name.lower() or 'ankle' in name.lower()]

    # Step 3: Find the minimum Y value across all foot joints and all frames (= ground level)
    foot_y_values = []
    for frame_poses in all_poses3d:
        if len(frame_poses) > 0:
            # First person only (consistent with TRC export)
            foot_y_values.append(frame_poses[0][foot_joints, 1].min())

    if foot_y_values:
        ground_level = np.percentile(foot_y_values, 5)  # 5th percentile to be robust to noise
        for i in range(len(all_poses3d)):
            if len(all_poses3d[i]) > 0:
                all_poses3d[i][:, :, 1] -= ground_level

    return all_poses3d


def save_to_json(filepath, all_poses3d, all_poses2d, all_confidences, joint_names, fps):
    """Sauvegarde toutes les données dans un seul fichier JSON lisible."""
    data = {
        "fps": fps,
        "joint_names": joint_names.tolist(),
        "frames": []
    }
    for i, (p3d, p2d, conf) in enumerate(zip(all_poses3d, all_poses2d, all_confidences)):
        frame_data = {
            "frame": i + 1,
            "time": round(i / fps, 5),
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


def save_to_trc(filepath, poses3d, joint_names, fps=30.0):
    """Sauvegarde les poses au format TRC (Track Row Column) pour la biomécanique (ex: OpenSim)."""
    num_frames = len(poses3d)
    num_markers = len(joint_names)

    with open(filepath, 'w') as f:
        # En-tête TRC (Lignes 1 à 6)
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tmm\t{fps}\t1\t{num_frames}\n")

        # Noms des marqueurs (Ligne 4)
        f.write("Frame#\tTime\t")
        for name in joint_names:
            f.write(f"{name}\t\t\t")
        f.write("\n")

        # Axes X, Y, Z (Ligne 5)
        f.write("\t\t")
        for _ in range(num_markers):
            f.write("X\tY\tZ\t")
        f.write("\n\n")

        # Données frame par frame
        for i, frame_poses in enumerate(poses3d):
            t = i / fps
            f.write(f"{i+1}\t{t:.5f}\t")

            # On extrait la première personne détectée. Si personne, des zéros.
            pose = frame_poses[0] if len(frame_poses) > 0 else np.zeros((num_markers, 3))

            for j in range(num_markers):
                x, y, z = pose[j]
                f.write(f"{x:.3f}\t{y:.3f}\t{z:.3f}\t")
            f.write("\n")


if __name__ == '__main__':
    main()
