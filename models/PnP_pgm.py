import torch
from torch.utils.data import Subset
import os 

from tqdm import tqdm
import sigpy as sp
from dataset.qmri_data_loader import MRIEchoDataset, get_mask
from util.util_algs import *
import yaml
from torch.utils.data import DataLoader, Subset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
def run_experiment():
    model_path = "model.pth"
    data_root_dir = "/files"
    test_subjects = ["S012"]
    batch_size = 1
    workers = 1
    simulated = False
    device, output_dir = setup_environment()
    alg_type = "pgm_tv"
    output_dir = "./results"
    save_outputs = True  # Set True to save image outputs
    config = load_config()
    mode = 'test'
    dataset = MRIEchoDataset(
        root_dir=config["data"]["root_dir_ksp"],
        sens_dir=config["data"]["root_dir_smps"],
        subjects=config[mode]["subjects"],
        slice_idxs=range(*config[mode]["slice_idxs"]),
        echos=range(*config[mode]["echos"]),
        multi_echo=config["general"]["multi_echo"],
    )

    if config["general"]["if_subset"]:
        dataset_sample = Subset(dataset, list(range(config["general"]["subset_len"])))
    else:
        dataset_sample = dataset
    test_loader = DataLoader(dataset_sample,batch_size=config[mode]["batch_size"],num_workers=config[mode]["num_workers"],pin_memory=False,persistent_workers=True,)

    # model = load_model(model_path, device)

    psnr_scores, ssim_in, in_psnr_score, ssim_out = [], [], [], []
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, samples in enumerate(tqdm(test_loader, desc="Testing")):
            ksp_gt, ksp_4x_real, smps_gt, smps_4x = samples
            ksp_normed, _, norm_factor = normalize_mri_data(ksp_gt, method=config["train"]["norm_type"])
            mask = get_mask().to(device)
            mask = mask[None, None, :, :, None].expand(ksp_gt.shape)

            ksp_normed = ksp_normed.to(device)
            ksp_gt = ksp_gt.to(device)
            ksp_4x_real = ksp_4x_real.to(device)
            smps_gt = smps_gt.to(device)
            smps_4x = smps_4x.to(device)

            ksp_4x = ksp_normed * mask
            psd_inv = pseduo_inverse(ksp_4x, smps_4x, mask)
            gt_img = multi_echo_transpose_operator(ksp_normed, smps_gt)
            zero_field = multi_echo_transpose_operator(ksp_4x, smps_4x)

            if alg_type == "sense":
                espirit_app = sp.mri.app.EspiritCalib(torch.view_as_complex(ksp_4x).squeeze().cpu().numpy(), device=0, show_pbar=False)
                sens_maps = espirit_app.run().get()  # shape: [coils, h, w]
                output = sp.mri.app.SenseRecon(torch.view_as_complex(ksp_4x).squeeze().cpu().numpy(), sens_maps, device=0, show_pbar = False).run().get()
                output = torch.tensor(output).to(ksp_4x.device).unsqueeze(0).unsqueeze(0)
                img_show(output, title="pgm")

            elif alg_type == "pgm_tv":
                output, psnr_iters = prox_grad_recon(torch.view_as_complex(ksp_4x),  smaps=torch.view_as_complex(smps_4x), mask=mask[...,0], img_gt=torch.view_as_complex(gt_img))
            else:
                print("Alg type is undefined")

            psnr = compare_psnr_batch(
                normalize_batch(torch.abs(torch.view_as_complex(gt_img))),
                normalize_batch(torch.abs(output)),
            ).mean().item()

            img_show(output, title="pgm")
            print("psnr pgm_tv", psnr)
            print("psnr psd_inv", compare_psnr_batch(
                normalize_batch(torch.abs(torch.view_as_complex(gt_img))),
                normalize_batch(torch.abs(torch.view_as_complex(psd_inv))),
            ).mean().item())
            img_show(torch.view_as_complex(gt_img), title="gt")
            img_show(torch.view_as_complex(psd_inv), title="syd")
            exit(90)


if __name__ == "__main__":
    run_experiment()





