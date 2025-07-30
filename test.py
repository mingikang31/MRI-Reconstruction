

# My Models 
from models.deep_unfolding_network import RUnet

from torch.utils.data import Subset, DataLoader
from utilities import to_complex, compare_psnr, compare_ssim
from utilities import * 
from dataset import MRIEchoDataset, get_mask


from utils.util import *
from utils.measures import *

### Visualization
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import subprocess
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from datetime import datetime
import glob

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def to_numpy(t):
    return torch.tensor(t).cpu().numpy() if isinstance(t, list) else t.cpu().numpy()

# def save_colorbar(filename='colorbar.png', vmin=0, vmax=255, cmap='magma'):
#     fig, ax = plt.subplots(figsize=(1.0, 4))  # (width, height) in inches
#     fig.subplots_adjust(left=0.5, right=0.8)
#     norm = plt.Normalize(vmin=vmin, vmax=vmax)
#     cbar = plt.colorbar(
#         plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#         cax=ax
#     )
#     cbar.set_label("Absolute Pixel Error", rotation=90)
#     plt.savefig(filename, dpi=150, bbox_inches='tight')
#     plt.close()

# save_colorbar("./results/colorbar.png", vmin=0, vmax=255)

import numpy as np
from PIL import Image

def normlize(data):
    """
    0-1 normlization
    Args:
        data: The input tensor
    Returns:
        The 0-1 normlized data.
    """
    return (data-data.min())/(data.max()-data.min())


def normalize_with_percentile(arr, percentile=98.0):
    """
    Normalizes an array by clipping the highlights to a given percentile.
    This increases brightness and contrast more aggressively.
    
    Args:
        arr (np.ndarray): The input NumPy array.
        percentile (float): The percentile to use for the maximum brightness.
    """
    vmin = arr.min()
    # Find the value at the given percentile
    vmax = np.percentile(arr, percentile)

    # Clip the array to this new min/max range
    clipped_arr = np.clip(arr, vmin, vmax)

    # Avoid division by zero
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)

    # Normalize the clipped array
    normalized_arr = (clipped_arr - vmin) / (vmax - vmin)
    
    # Scale to 0-255 and convert to uint8
    return (normalized_arr * 255.0).astype(np.uint8)

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


# --------- PARAMETERS ---------
# model_path = "/export1/project/mingi/qmri/model_zoo/RUnet_Full_Dataset_GradientLoss/Val_N2N_150.pth" 
model_path = "/export1/project/mingi/MRI-Reconstruction/model_save/RUnet/Val_N2N_150.pth" 


data_root_dir = "/project/cigserver5/export1/shirin/Dataset/qmri/pt_files"
test_subjects = ["S012"]  # Change or add more if needed
batch_size = 1
workers = 1
save_outputs = True  # Set True to save image outputs
now = datetime.now()
model_name = "RUnet_Full_Dataset_GradientLoss"
output_dir = "/export1/project/mingi/qmri/results/Full/"+model_name
simulated = True
# ------------------------------

print("model_path:", model_path)
print("output_dir:", output_dir)

# Load dataset
test_dataset = MRIEchoDataset(h5_dir=data_root_dir, subjects=test_subjects)
# Limit dataset to first 40 samples
subset_dataset = Subset(test_dataset, range(140))

# Create new DataLoader with the subset
testloader_100 = DataLoader(
    subset_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=False
)
    
# Load model
model = RUnet(pools=4, chans=64).to(device) # RUnet

model.load_state_dict(torch.load(model_path))
model.eval()

psnr_scores, ssim_in , in_psnr_score , ssim_out= [], [], [], []

os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for i, samples in enumerate(tqdm(testloader_100, desc="Testing")):

        # Prepare data
        ksp_gt = to_complex(samples['ksp_gt'].float(), device)
        smps_gt, gt_img = walsh_sensitivity_maps(torch.view_as_real(ksp_gt))

        norm_factor_gt = ksp_gt.abs().max()
        gt_img = gt_img / norm_factor_gt
        ksp_gt = ksp_gt / norm_factor_gt
        mask = get_mask().to(device)
        
        if simulated:
            smps = to_complex(samples['sens_map'].float(), device)
            ksp_4x = (ksp_gt * mask[None, None, :, :])
            norm_factor_4x = ksp_4x.abs().max()
            input_img = zero_filled_rss(torch.view_as_real(ksp_4x/norm_factor_4x))
            ksp_4x = ksp_4x / norm_factor_4x
        else: 
            ksp_4x = to_complex(samples['ksp_4x'].float(), device)
            smps = to_complex(samples['sens_map'].float(), device)
            norm_factor_4x = ksp_4x.abs().max()
            input_img = zero_filled_rss(torch.view_as_real(ksp_4x/norm_factor_4x))
            ksp_4x = ksp_4x / norm_factor_4x
            

        # Output img 
        output_img, smps = model(torch.view_as_real(ksp_4x), mask)

        output_mag = normlize(torch.abs(torch.view_as_complex(output_img)))
        target_mag = normlize(torch.abs(torch.view_as_complex(gt_img)))
        input_mag = normlize(torch.abs(input_img))

        in_psnrs = compare_psnr_batch(target_mag, input_mag)
        psnrs = compare_psnr_batch(target_mag, output_mag)
        psnr_scores.extend(psnrs.cpu().tolist())
        in_psnr_score.extend(in_psnrs.cpu().tolist())

        ssim_out.append(compare_ssim(output_mag, target_mag))
        ssim_in.append(compare_ssim(input_mag, target_mag))

        echo = i//70 +1
        if save_outputs:
            os.makedirs(output_dir + f"/echo_{echo}/gt", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/in", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/out", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/err_out", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/err_in", exist_ok=True)
            for j in range(output_img.shape[0]):

                gt_np = target_mag[j, ...].squeeze().cpu().numpy()
                rss_uint8_gt = (np.array(gt_np) * 255.0).astype(np.uint8)
                rss_uint8_gt = normalize_with_percentile(rss_uint8_gt, percentile=98.0)  # Normalize with percentile
                img = Image.fromarray(rss_uint8_gt)
                img.save(output_dir +f"/echo_{echo}/gt/img_{i%70}.png")

                out_np = output_mag[j].squeeze().cpu().numpy()
                rss_uint8_out = (np.array(out_np) * 255.0).astype(np.uint8)
                rss_uint8_out = normalize_with_percentile(rss_uint8_out, percentile=98.0)  # Normalize with percentile
                img = Image.fromarray(rss_uint8_out)
                img.save(output_dir + f"/echo_{echo}/out/img_{i%70}.png")

                in_np = input_mag[j].squeeze().cpu().numpy()
                rss_uint8_in = (np.array(in_np) * 255.0).astype(np.uint8)
                rss_uint8_in = normalize_with_percentile(rss_uint8_in, percentile=98.0)  # Normalize with percentile
                img = Image.fromarray(rss_uint8_in)
                img.save(output_dir + f"/echo_{echo}/in/img_{i%70}.png")

                err_in = np.abs(gt_np - in_np)
                colormap = plt.get_cmap('magma')  # choose any: 'gray', 'jet', 'magma', etc.
                colored_arr = colormap(err_in)[:, :, :3]  # drop alpha channel if exists
                colored_arr = (colored_arr * 255).astype(np.uint8)
                img = Image.fromarray(colored_arr)
                img.save(output_dir + f"/echo_{echo}/err_in/img_{i % 70}.png")

                err_out = np.abs(gt_np - out_np)
                colormap = plt.get_cmap('magma')  # choose any: 'gray', 'jet', 'magma', etc.
                colored_arr = colormap(err_out)[:, :, :3]  # drop alpha channel if exists
                colored_arr = (colored_arr * 255).astype(np.uint8)
                img = Image.fromarray(colored_arr)
                img.save(output_dir + f"/echo_{echo}/err_out/img_{i%70}.png")




# csv_path = os.path.join(output_dir, "psnr_results.csv")
# with open(csv_path, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Slice_Index", "PSNR_ZF", "PSNR_Model", "SSIM_ZF", "SSIM_Model"])
#     for idx, (psnr_m, psnr_b, ssim_a, ssim_b) in enumerate(
#             zip(to_numpy(psnr_scores), to_numpy(in_psnr_score),
#                 to_numpy(ssim_out), to_numpy(ssim_in))):
#         writer.writerow([idx, f"{psnr_b:.4f}", f"{psnr_m:.4f}", f"{ssim_b:.4f}", f"{ssim_a:.4f}"])

# print(f"\n📄 PSNR values saved to {csv_path}")
print(f"📊 Avg PSNR (Model): {np.mean(to_numpy(psnr_scores)):.2f} | Avg PSNR (ZF): {np.mean(to_numpy(in_psnr_score)):.2f}")

print(f"📊 Avg SSIM (Model): {np.mean(to_numpy(ssim_out)):.4f} | Avg SSIM (ZF): {np.mean(to_numpy(ssim_in)):.4f}")



# Video generation 
models = ["RUnet_Full_Dataset_GradientLoss"] # ["UNet_S_Sim", "AttentionUNet_S_Sim", "SPNet_S_Sim"]
echos = ["echo_1", "echo_2"]
subdir = ["err_in", "err_out", "gt", "in", "out"]

# for model in models:
#     for echo in echos:
#         for sub in subdir:
#             print(f"Generating video for {model} - {echo} - {sub}")
#             video(model, echo, sub)

# # scp -r "mingi@cigserver5.engr.wustl.edu:/export1/project/mingi/qmri/results/Full/RUnet_Full_Dataset_GradientLoss_Real" "/Users/mingikang/Desktop/qmri/Full/"


