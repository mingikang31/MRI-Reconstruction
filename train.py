import torch
from torch.nn import functional as F

# from basic_network import VarNet
from networks.deep_unfolding_network import SPNet, NormUnet, Unet, RUnet
from tqdm import tqdm
from datetime import datetime
from dataset import MRIEchoDataset, get_mask
import torch.optim as optim
from utils.util import *
from utils.measures import *
from utils.loss_functions import *
scaler = torch.cuda.amp.GradScaler()
from torch.utils.data import Subset
import wandb
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as ssim


if torch.cuda.is_available():
    device = 'cuda:2'
else:
    device = 'cpu'

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
simulated = True 

from piq import SSIMLoss


### Weighted weight map 
def create_gaussian_weight_map(height=234, width=176, sigma=40):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    center_y, center_x = height // 2, width // 2
    dist_sq = (x - center_x)**2 + (y - center_y)**2
    weight_map = 1.0 + 10.0 * torch.exp(-dist_sq / (2 * sigma**2))
    return weight_map.to(device)

weight_map = create_gaussian_weight_map()

def gradient_loss_img(output, gt): 
    sobel_kernel_x = torch.tensor([[1, 0, -1],
                                   [2, 0, -2],
                                   [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)  

    output_grad_x = F.conv2d(output, sobel_kernel_x, padding=1)
    output_grad_y = F.conv2d(output, sobel_kernel_y, padding=1)
    gt_grad_x = F.conv2d(gt, sobel_kernel_x, padding=1)
    gt_grad_y = F.conv2d(gt, sobel_kernel_y, padding=1)

    grad_loss_x = F.mse_loss(output_grad_x, gt_grad_x)
    grad_loss_y = F.mse_loss(output_grad_y, gt_grad_y)
    return grad_loss_x + grad_loss_y


def train(epoch):
    model.train()
    train_av_epoch_psnr_list = []
    train_av_epoch_loss_list = []
    train_av_epoch_ssim_list = []
    for iteration, samples in enumerate(iter_):
        ksp_gt = to_complex(samples['ksp_gt'].float(), device)
        smps_gt , gt_img = walsh_sensitivity_maps(torch.view_as_real(ksp_gt))
        norm_factor = gt_img.abs().max()
        gt_img = gt_img/norm_factor
        mask = get_mask().to(device)
        ksp_4x = (ksp_gt * mask[None, None, :, :])/norm_factor
        input_img = zero_filled_rss(torch.view_as_real(ksp_4x))
        with torch.cuda.amp.autocast():
            output_img, smps= model(torch.view_as_real(ksp_4x), mask)

            recons_ksp = grad_g_operator(out_img=output_img, smap=smps)
            loss_ksp = F.mse_loss(
                torch.view_as_real((ksp_gt/norm_factor)), 
                recons_ksp
            )

            # Losses 
            mse_loss = F.mse_loss(torch.view_as_complex(output_img).abs(), torch.view_as_complex(gt_img).abs())
            gradient_loss_value = gradient_loss_img(torch.view_as_complex(output_img).abs(), torch.view_as_complex(gt_img).abs())
            smps_loss = gradient_loss(smps)
            loss = 1.5 * mse_loss + 0.5 * loss_ksp + 0.001 * smps_loss + 0.1 * gradient_loss_value

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_av_epoch_psnr_list.append(np.mean(compare_psnr_batch(normalize_batch(torch.abs(torch.view_as_complex(gt_img))), normalize_batch(torch.abs(torch.view_as_complex(output_img)))).cpu().detach().numpy()))
        train_av_epoch_loss_list.append(loss.item())

    if epoch % 3 == 0:
        target_show = normlize(torch.abs(torch.view_as_complex(gt_img[0,...])).to('cpu').detach())
        output_show = normlize(torch.abs(torch.view_as_complex(output_img[0, ...])).to('cpu').detach())
        input_show  = normlize(input_img[0, ...].to('cpu').detach())
        concat_img = np.concatenate([input_show, output_show, target_show], axis=2)
        wandb.log({
            "Train/concat_image": wandb.Image(concat_img, caption="Input | Output | Target")
        })

    psnr_value = np.mean(np.array(train_av_epoch_psnr_list))
    loss_value = np.mean(train_av_epoch_loss_list)


    wandb.log({
        "train/psnr": psnr_value,
        "train/loss": loss_value,
        "epoch": epoch
    })
    print('The PSNR value for  output is {}'.format(psnr_value))
    # print('The SSIM value for N2N output is {}'.format(ssim_value))
    print('training loss at epoch {} is {}'.format(epoch, loss_value))


    # Saving logic
    if epoch % 5 == 0 and epoch > 0:
        print(f'save the model at epoch {epoch}')
        model_dir = f'/export1/project/mingi/qmri/model_zoo/{model_name}'
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{model_dir}/Train_N2N_{epoch:03d}.pth")

def val(epoch):
    model.eval()
    eval_av_epoch_psnr_list = []
    eval_av_epoch_loss_list = []

    for iteration, samples in enumerate(valloader):
        ### Original 
        ksp_gt = to_complex(samples['ksp_gt'].float(), device)
        smps_gt , gt_img = walsh_sensitivity_maps(torch.view_as_real(ksp_gt))
        norm_factor = gt_img.abs().max()
        gt_img = gt_img/norm_factor
        mask = get_mask().to(device)
        ksp_4x = (ksp_gt * mask[None, None, :, :])/norm_factor
        input_img = zero_filled_rss(torch.view_as_real(ksp_4x))
        
        output_img, smps = model(torch.view_as_real(ksp_4x), mask)

        recons_ksp = grad_g_operator(out_img=output_img, smap=smps)

        # ### Original Losses
        # loss_ksp = F.mse_loss(
        #     torch.view_as_real((ksp_gt/norm_factor)), 
        #     recons_ksp
        # )
        # mse_loss = F.mse_loss(torch.view_as_complex(output_img).abs(), torch.view_as_complex(gt_img).abs())
        # smps_loss = gradient_loss(smps)
        # loss = 1.5 * mse_loss + 0.5 * loss_ksp + 0.001 * smps_loss



        # ### Weighted Map Losses
        # ## Add weighted map to mse loss
        # loss_ksp = F.mse_loss(
        #     torch.view_as_real((ksp_gt/norm_factor)), 
        #     recons_ksp
        # )
        # mse_loss = F.mse_loss(torch.view_as_complex(output_img).abs(), torch.view_as_complex(gt_img).abs(), reduction='none')
        # weighted_loss = mse_loss * weight_map[None, None, :, :]
        # mse_loss = weighted_loss.mean()

        # Gradient Losses 
        loss_ksp = F.mse_loss(
            torch.view_as_real((ksp_gt/norm_factor)), 
            recons_ksp
        )
        mse_loss = F.mse_loss(torch.view_as_complex(output_img).abs(), torch.view_as_complex(gt_img).abs())
        gradient_loss_value = gradient_loss_img(torch.view_as_complex(output_img).abs(), torch.view_as_complex(gt_img).abs())
        smps_loss = gradient_loss(smps)
        loss = 1.5 * mse_loss + 0.5 * loss_ksp + 0.001 * smps_loss + 0.1 * gradient_loss_value

        eval_av_epoch_psnr_list.append(np.mean(compare_psnr_batch(normalize_batch(torch.abs(torch.view_as_complex(gt_img))), normalize_batch(torch.abs(torch.view_as_complex(output_img)))).cpu().detach().numpy()))
        eval_av_epoch_loss_list.append(loss.item())

    target_show = normlize(torch.abs(torch.view_as_complex(gt_img[0, ...])).to('cpu').detach())
    output_show = normlize(torch.abs(torch.view_as_complex(output_img[0, ...])).to('cpu').detach())
    input_show = normlize(input_img[0, ...].to('cpu').detach())
    concat_img = np.concatenate([input_show, output_show, target_show], axis=2)
    wandb.log({
        "val/concat_image": wandb.Image(concat_img, caption="Input | Output | Target")
    })

    psnr_value = np.mean(np.array(eval_av_epoch_psnr_list))
    loss_value = np.mean(eval_av_epoch_loss_list)
    wandb.log({
        "val/psnr": psnr_value,
        "val/loss": loss_value,
        "epoch": epoch
    })
    print(f'val: The PSNR value for output is {psnr_value:.4f}')
    print(f'val: Validation loss at epoch {epoch} is {loss_value:.6f}')

    snr_best.append(psnr_value)

    # Saving logic
    if epoch % 5 == 0 and epoch > 0:
        if psnr_value >= np.max(np.array(snr_best)):
            print(f'save the model at epoch {epoch}')
            model_dir = f'/export1/project/mingi/qmri/model_zoo/{model_name}'
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{model_dir}/Val_N2N_{epoch:03d}.pth")

if __name__ == "__main__":
    # ------------
    # parameters
    # ------------
    now = datetime.now()
    model_name = 'RUNET_KAIMING_UNIFORM'
    batch = 16
    workers = 4
    epoch_number = 400
    acceleration_factor = 4
    save_root = f'/export1/project/mingi/qmri/model_zoo/{model_name}'
    if not (os.path.exists(save_root)): os.makedirs(save_root)

    USE_WANDB = True
    if USE_WANDB:
        wandb.init(project="UNet_Denoiser", name=f"RUnet Run-{now.strftime('%m%d-%H%M%S')}", config={
            "epochs": epoch_number,
            "batch_size": batch,
            "lr": 0.001,
            "acceleration_factor": acceleration_factor,
            "model": "RUnet",
            "device": device,
        }) 

    train_subject = ["S001", "S003", "S004", "S005", "S006", "S007", "S010"]
    val_subject = ["S012"]
    data_root_dir ="/project/cigserver5/export1/shirin/Dataset/qmri/pt_files"

    train_dataset = MRIEchoDataset(h5_dir=data_root_dir,subjects=train_subject)
    print(f"Total training samples: {len(train_dataset)}")
    val_dataset = MRIEchoDataset(h5_dir=data_root_dir,subjects=val_subject)
    print(f"Total validation samples: {len(val_dataset)}")

    val_dataset = Subset(val_dataset, list(range(1))) # For debugging purposes
    valloader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)

    train_dataset = Subset(train_dataset, list(range(1))) # For debugging purposes
    trainloader = DataLoader(train_dataset, batch_size=batch, num_workers=workers, pin_memory=False,persistent_workers=True)


    model = RUnet(pools=4, chans=64).to(device)
    model.apply(lambda x: torch.nn.init.kaiming_uniform_(x.weight) if hasattr(x, 'weight') and x.weight is not None else None) 
    print(f"Total number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    snr_best = []
    for epoch in range(epoch_number):
 
        iter_ = tqdm(trainloader, desc='Train [%.3d/%.3d]' % (epoch, epoch_number), total=len(trainloader))
        train(epoch)
        with torch.no_grad():
            val(epoch)

        lr_scheduler(optimizer, epoch)
