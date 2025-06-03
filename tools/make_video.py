import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
# import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--fps', type=int, default=3)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=-1)
parser.add_argument('--output', type=str, default='output.mp4')
args = parser.parse_args()

base_dir = 'vis_result'
camera_dirs = [os.path.join(base_dir, f'camera-{i}') for i in range(6)]
lidar_dir = os.path.join(base_dir, 'lidar')
# output_gif = 'output.gif'

frame_names = sorted(os.listdir(camera_dirs[0]))[args.start : (args.end if args.end > 0 else None)]

frames = []

for name in tqdm(frame_names):

    cam_imgs = []
    for cam_dir in camera_dirs:
        img_path = os.path.join(cam_dir, name)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Missing image: {img_path}")
        cam_imgs.append(img)

    h, w = cam_imgs[0].shape[:2]
    cam_imgs = [cv2.resize(img, (w, h)) for img in cam_imgs]

    top_row = np.hstack([cam_imgs[2], cam_imgs[0], cam_imgs[1]])
    bottom_row = np.hstack([cam_imgs[5], cam_imgs[3], cam_imgs[4]])
    camera_grid = np.vstack([top_row, bottom_row])

    lidar_path = os.path.join(lidar_dir, name)
    lidar_img = cv2.imread(lidar_path)
    if lidar_img is None:
        raise ValueError(f"Missing lidar image: {lidar_path}")
    lidar_img = cv2.resize(lidar_img, (w, h))
    empty = np.zeros_like(lidar_img)
    lidar_row = np.hstack([empty, lidar_img, empty])

    combined = np.vstack([camera_grid, lidar_row])

    frames.append(combined)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size = (int(frames[0].shape[1] * 0.5), int(frames[0].shape[0] * 0.5))
out = cv2.VideoWriter(args.output, fourcc, args.fps, frame_size)
for f in tqdm(frames):
    f = cv2.resize(f, frame_size)
    out.write(f)
out.release()

print(f"Video saved to {args.output}")

# gif_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
# imageio.mimsave(output_gif, gif_frames, fps=5)

# print(f"GIF saved to {output_gif}")
