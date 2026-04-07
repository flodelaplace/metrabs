import sys
import urllib.request

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

    all_poses3d = []  # Liste qui va stocker les poses de toute la vidéo

    if USE_POSEVIZ:
        viz = poseviz.PoseViz(joint_names, joint_edges)

    for i, frame_batch in enumerate(tqdm(frame_batches, total=total_batches, desc="Inférence 3D en cours", unit="batch")):
        pred = model.detect_poses_batched(
            frame_batch, intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
            skeleton=skeleton)
        
        # On extrait les poses image par image dans le batch
        for frame, boxes, poses in zip(frame_batch, pred['boxes'], pred['poses3d']):
            # poses.numpy() est un tableau de taille (Nb_personnes, 87, 3)
            all_poses3d.append(poses.numpy())
            
            if USE_POSEVIZ:
                viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)

    if USE_POSEVIZ:
        viz.close()

    # Sauvegarde finale
    output_filename = "poses3d_output.trc"
    save_to_trc(output_filename, all_poses3d, joint_names, fps)
    print(f"\nTerminé ! Les coordonnées 3D sont sauvegardées dans : {output_filename}")


def get_video(source, temppath='/tmp/video.mp4'):
    if not source.startswith('http'):
        return source

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(source, temppath)
    return temppath

def save_to_trc(filepath, poses3d, joint_names, fps=30.0):
    """Sauvegarde les poses au format TRC (Track Row Column) pour la biomécanique (ex: OpenSim)."""
    num_frames = len(poses3d)
    num_markers = len(joint_names)
    
    with open(filepath, 'w') as f:
        # En-tête TRC (Lignes 1 à 3)
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
