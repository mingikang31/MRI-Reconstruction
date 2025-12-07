"""
Testing Script to save Images and compute PSNR/SSIM metrics
"""
import os 

import torch 
from torch.utils.data import DataLoader, Subset
import numpy as np 
from datetime import datetime
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import csv

from models.recon_unet import ReconUnet
from models.recon_attention_unet import ReconAttentionUnet
from visualize import video, normalize_with_percentile, to_numpy, normalize
from utils.util import walsh_sensitivity_maps, zero_filled_rss, to_complex, get_mask
from utils.metric import compare_psnr_batch, compare_ssim
from dataset import MRIEchoDataset

# --------- PARAMETERS ---------
model_path = "EXAMPLE_MODEL_PATH.pth"  # Path to the trained model

data_root_dir = "DATA_DIR_PATH"  # Directory containing test data
test_subjects = ["S012"]  # Change or add more if needed
batch_size = 1
workers = 1
save_outputs = True  # Set True to save image outputs
now = datetime.now()
model_name = "RUnet"
output_dir = "./test_outputs/" + model_name + "_" + now.strftime("%Y%m%d_%H%M%S")
simulated = True  # Set to True to match your original code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------------

print("model_path:", model_path)
print("output_dir:", output_dir)

# Load dataset
test_dataset = MRIEchoDataset(h5_dir=data_root_dir, subjects=test_subjects, train=False)
subset_dataset = Subset(test_dataset, range(140))

# Load DataLoader 
test_loader = DataLoader(
    subset_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=workers,
    pin_memory=False
)

# Load Model 
model = ReconUnet(channels=64, pool_layers=4).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Metrics (matching your original variable names)
psnr_scores, ssim_in, in_psnr_score, ssim_out = [], [], [], []

os.makedirs(output_dir, exist_ok=True)

with torch.no_grad():
    for i, samples in enumerate(tqdm(test_loader, desc="Testing")):
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

        # Forward pass
        output_img, smps = model(torch.view_as_real(ksp_4x), mask)

        # Get magnitudes and normalize using your original method
        output_mag = normalize(torch.abs(torch.view_as_complex(output_img)))
        target_mag = normalize(torch.abs(torch.view_as_complex(gt_img)))
        input_mag = normalize(torch.abs(input_img))

        in_psnrs = compare_psnr_batch(target_mag, input_mag)
        psnrs = compare_psnr_batch(target_mag, output_mag)
        psnr_scores.extend(psnrs.cpu().tolist())
        in_psnr_score.extend(in_psnrs.cpu().tolist())

        ssim_out.append(compare_ssim(output_mag, target_mag))
        ssim_in.append(compare_ssim(input_mag, target_mag))

        echo = i//70 + 1
        if save_outputs:
            os.makedirs(output_dir + f"/echo_{echo}/gt", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/in", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/out", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/err_out", exist_ok=True)
            os.makedirs(output_dir + f"/echo_{echo}/err_in", exist_ok=True)
            
            for j in range(output_img.shape[0]):
                gt_np = target_mag[j, ...].squeeze().cpu().numpy()
                rss_uint8_gt = (np.array(gt_np) * 255.0).astype(np.uint8)
                rss_uint8_gt = normalize_with_percentile(rss_uint8_gt, percentile=98.0)
                img = Image.fromarray(rss_uint8_gt)
                img.save(output_dir + f"/echo_{echo}/gt/img_{i%70}.png")

                out_np = output_mag[j].squeeze().cpu().numpy()
                rss_uint8_out = (np.array(out_np) * 255.0).astype(np.uint8)
                rss_uint8_out = normalize_with_percentile(rss_uint8_out, percentile=98.0)
                img = Image.fromarray(rss_uint8_out)
                img.save(output_dir + f"/echo_{echo}/out/img_{i%70}.png")

                in_np = input_mag[j].squeeze().cpu().numpy()
                rss_uint8_in = (np.array(in_np) * 255.0).astype(np.uint8)
                rss_uint8_in = normalize_with_percentile(rss_uint8_in, percentile=98.0)
                img = Image.fromarray(rss_uint8_in)
                img.save(output_dir + f"/echo_{echo}/in/img_{i%70}.png")

                err_in = np.abs(gt_np - in_np)
                colormap = plt.get_cmap('magma')
                colored_arr = colormap(err_in)[:, :, :3]
                colored_arr = (colored_arr * 255).astype(np.uint8)
                img = Image.fromarray(colored_arr)
                img.save(output_dir + f"/echo_{echo}/err_in/img_{i % 70}.png")

                err_out = np.abs(gt_np - out_np)
                colormap = plt.get_cmap('magma')
                colored_arr = colormap(err_out)[:, :, :3]
                colored_arr = (colored_arr * 255).astype(np.uint8)
                img = Image.fromarray(colored_arr)
                img.save(output_dir + f"/echo_{echo}/err_out/img_{i%70}.png")

# Save results to CSV
csv_path = os.path.join(output_dir, "psnr_results.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Slice_Index", "PSNR_ZF", "PSNR_Model", "SSIM_ZF", "SSIM_Model"])
    for idx, (psnr_m, psnr_b, ssim_a, ssim_b) in enumerate(
            zip(to_numpy(psnr_scores), to_numpy(in_psnr_score),
                to_numpy(ssim_out), to_numpy(ssim_in))):
        writer.writerow([idx, f"{psnr_b:.4f}", f"{psnr_m:.4f}", f"{ssim_b:.4f}", f"{ssim_a:.4f}"])

print(f"\n📄 PSNR values saved to {csv_path}")
print(f"📊 Avg PSNR (Model): {np.mean(to_numpy(psnr_scores)):.2f} | Avg PSNR (ZF): {np.mean(to_numpy(in_psnr_score)):.2f}")
print(f"📊 Avg SSIM (Model): {np.mean(to_numpy(ssim_out)):.4f} | Avg SSIM (ZF): {np.mean(to_numpy(ssim_in)):.4f}")

# Video Generation 
echos = ["echo_1", "echo_2"]
subdir = ["err_in", "err_out", "gt", "in", "out"]

for echo in echos:
    for sub in subdir:
        print(f"Generating video for {model_name} - {echo} - {sub}")
        video(output_dir, sub, echo)