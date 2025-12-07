"""
Visualization utilities for MRI reconstruction results.
"""
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from tqdm import tqdm
from datetime import datetime 
import os 
import subprocess 
import glob 
import torch 

def save_colorbar(filename='colorbar.png', vmin=0, vmax=255, cmap='magma'):
    """Create and save a colorbar for visualizations."""
    fig, ax = plt.subplots(figsize=(1.0, 4))  # (width, height) in inches
    fig.subplots_adjust(left=0.5, right=0.8)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax
    )
    cbar.set_label("Absolute Pixel Error", rotation=90)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()    

def video(dir, subdir, echo):
    """Generate Video from saved PNG frames using ffmpeg."""
    path = f"{dir}/{echo}/{subdir}"

    if not os.path.exists(path):
        print(f"Path {path} does not exist. Skipping video generation.")
        return

    output_dir = f'{dir}/output_videos/{echo}/'
    temp_dir = f'{dir}/frames/{echo}/{subdir}'

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    fps = 10

    filenames = glob.glob(os.path.join(path, '**', '*.png'), recursive=True)
    print(filenames)
    for i, f in enumerate(tqdm(filenames, total=len(filenames))):
        img1 = Image.open(os.path.join(path, f))

        if "err" in path:
            img_array = np.array(img1).astype(np.float32)
            scaled_array = img_array * 5  # for example
            scaled_array = np.clip(scaled_array, 0, 255).astype(np.uint8)
            img1 = Image.fromarray(scaled_array)

        if img1.mode == 'L':
            rgb = Image.merge('RGB', (img1, img1, img1))
        else:
            rgb = img1

        rgb.save(os.path.join(temp_dir, f"frame_{i:04d}.png"))

    output_video_path = os.path.join(output_dir, f'{subdir}.mp4')
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # overwrite without asking
        '-framerate', str(fps),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-c:v', 'mpeg4',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    print("🛠️ Running ffmpeg...")
    subprocess.run(ffmpeg_cmd, check=True)
    print("✅ Video created:", output_video_path)

def to_numpy(x):
    """Convert torch tensor to numpy array."""
    return torch.tensor(x).cpu().numpy() if isinstance(x, list) else x.cpu().numpy()

def normalize(data):
    """Normalize data to [0, 1]."""
    return (data-data.min())/(data.max()-data.min())


def normalize_with_percentile(arr, percentile=98.0):
    """ Normalizes an array by clipping the highlights to a given percentile. """
    vmin = arr.min()
    vmax = np.percentile(arr, percentile)

    clipped_arr = np.clip(arr, vmin, vmax)

    # Avoid division by zero
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)

    normalized_arr = (clipped_arr - vmin) / (vmax - vmin)
    return (normalized_arr * 255.0).astype(np.uint8)
