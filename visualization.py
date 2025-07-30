import os
from PIL import Image, ImageChops
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image, ImageChops
import subprocess
import re
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.io import savemat
from datetime import datetime
import glob


def save_colorbar(filename='colorbar.png', vmin=0, vmax=255, cmap='magma'):
    fig, ax = plt.subplots(figsize=(1.0, 4))  # (width, height) in inches
    fig.subplots_adjust(left=0.5, right=0.8)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax
    )
    cbar.set_label("Absolute Pixel Error", rotation=90)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# save_colorbar("./results/colorbar.png", vmin=0, vmax=255)

def get_psnr():
    path_a = "/project/cigserver5/export1/shirin/qmri/results/one/one.png"
    path_b = "/project/cigserver5/export1/shirin/qmri/results/two/two.png"

    img1 = Image.open(path_a).convert('L')
    img2 = Image.open(path_b).convert('L')
    arr1 = np.transpose(np.array(img1.transpose(Image.ROTATE_180)), (1,0))
    arr2 = np.array(img2)
    psnr = compare_psnr(arr1, arr2, data_range=255.0)

    diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16)).astype(np.uint8) / 255.0
    colormap = plt.get_cmap('magma')
    rgba_img = colormap(diff)
    diff_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)
    rgb = Image.fromarray(diff_img)
    rgb.save("./results/diff_image.png")
    print(psnr)

def video(model, echo, subdir):
    now = datetime.now()
    # name_folder = str(now.strftime("%d-%b-%Y-%H-%M-%S"))
    path = f"/export1/project/mingi/qmri/results/Full/{model}/{echo}/{subdir}"
    if not os.path.exists(path):
        print(f"Path {path} does not exist. Skipping video generation.")

    output_dir = f'/export1/project/mingi/qmri/results/Full/{model}/output_videos/{echo}/'
    temp_dir = f'/export1/project/mingi/qmri/results/Full/{model}/frames/{echo}/{subdir}'
    
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



if __name__ == "__main__":
    # Video generation 
    models = ["UNet_S_Sim", "AttentionUNet_S_Sim", "SPNet_S_Sim"]
    echos = ["echo_1", "echo_2"]
    subdir = ["err_in", "err_out", "gt", "in", "out"]

    for model in models:
        for echo in echos:
            for sub in subdir:
                print(f"Generating video for {model} - {echo} - {sub}")
                video(model, echo, sub)

    # Image + PSNR generation 


# scp "mingi@cigserver5.engr.wustl.edu:/export1/project/mingi/qmri/results/Full/AttentionUNet_S_Sim/output_videos/echo_1/*.mp4" /Users/mingikang/Desktop/qmri/Full/AttentionUNet_S_Sim/echo_1/

# scp "mingi@cigserver5.engr.wustl.edu:/export1/project/mingi/qmri/results/Full/UNet_S_Sim/output_videos/echo_1/*.mp4" /Users/mingikang/Desktop/qmri/Full/UNet_S_Sim/echo_1/  

# scp "mingi@cigserver5.engr.wustl.edu:/export1/project/mingi/qmri/results/Full/SPNet_S_Sim/output_videos/echo_1/*.mp4" /Users/mingikang/Desktop/qmri/Full/SPNet_S_Sim/echo_1/