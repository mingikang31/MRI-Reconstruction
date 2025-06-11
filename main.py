"""Main File for the project"""

import argparse 
from pathlib import Path
import os 

# Datasets 
from dataset import *

from UNet import * 
from AttentionUNet import * 
from ViT import * 
from FCN import * 

# SPICER UNet 
from UNet_SPICER import * 

from train import * 


def args_parser():
    parser = argparse.ArgumentParser(description="GRAPPA UNet Training", add_help=False) 
    
    # parser.add_argument("--acc", type=int, default=4, help="Acceleration factor for GRAPPA")   

    # parser.add_argument("--num_coils", type=int, default=8, help="Number of coils for GRAPPA")
    # parser.add_argument("--acs_lines", type=int, default=32, help="Number of autocalibration lines for GRAPPA")
    # parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size for GRAPPA")

    # parser.add_argument("--operator", type=str, default="GRAPPA", choices=["GRAPPA", "MRI"], help="Operator to use for reconstruction")
    

    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=300, help="Number of epochs for training")

    
    # # Loss Function Arguments
    # parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    
    # # Optimizer Arguments 
    # parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')

    
    # Learning Rate Arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")

    # Device Arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training and evaluation")
    parser.add_argument('--seed', default=0, type=int)
    
    # Output Arguments 
    parser.add_argument("--outdir", type=str, default="./Saved/UNet/", help="Directory to save the weights")


    ## Continue Training Arguments
    parser.add_argument("--continue_training", action="store_true", help="Continue training from the last checkpoint")
    parser.set_defaults(continue_training=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint to continue training from")
    return parser

    
def main(args):

    os.makedirs(args.outdir, exist_ok=True)  # Create output directory if it doesn't exist
    
    import sys
    log_file = open(args.outdir + 'training_output.txt', 'w')
    sys.stdout = type('', (), {
        'write': lambda self, text: [sys.__stdout__.write(text), log_file.write(text)],
        'flush': lambda self: [sys.__stdout__.flush(), log_file.flush()]
    })()


    # Load Dataset 
    IMAGES_DIR = "./data/4000_img/GRAPPA_acc2"
    RECONSTRUCTION_DIR = "./data/4000_img/GRAPPA_acc6"

    images = load_img_to_tensor(IMAGES_DIR, max_images=4000, device=args.device) # normalized + (320, 320) shape
    reconstructions = load_img_to_tensor(RECONSTRUCTION_DIR, max_images=4000, device=args.device) # normalized + (320, 320) shape

    train_dataset = ProcessedMRIDataset(images[:3200], reconstructions[:3200])
    test_dataset = ProcessedMRIDataset(images[3200:], reconstructions[3200:])

    train_dataloader = ProcessedMRIDataLoader(train_dataset,
                                        batch_size=args.batch_size, 
                                        shuffle=True, 
                                        )
    test_dataloader = ProcessedMRIDataLoader(test_dataset,
                                        batch_size=args.batch_size, 
                                        shuffle=False, 
                                        )
    
    
    # MRI_DIR = "/export1/project/mingi/Dataset/brainMRI"
    # images = load_img_to_tensor(MRI_DIR, max_images=8000, device=args.device) # normalized + (320, 320) shape 

    # train_images, test_images = split_images(images, train_ratio=0.8)

    # train_dataset = MRIDataset(train_images, 
    #                          acc=args.acc, 
    #                          num_coils=args.num_coils, 
    #                          acs_lines=args.acs_lines,
    #                          kernel_size=(args.kernel_size, args.kernel_size), 
    #                          operator=GRAPPAOperator if args.operator == "GRAPPA" else MRIOperator,
    #                          device=args.device)

    # test_dataset = MRIDataset(test_images,
    #                             acc=args.acc, 
    #                             num_coils=args.num_coils, 
    #                             acs_lines=args.acs_lines,
    #                             kernel_size=(args.kernel_size, args.kernel_size), 
    #                             operator=GRAPPAOperator if args.operator == "GRAPPA" else MRIOperator,
    #                             device=args.device)

    
    
    # train_dataloader = MRIDataLoader(train_dataset, 
    #                                batch_size=args.batch_size, 
    #                                shuffle=True, 
    #                                operator=GRAPPAOperator if args.operator == "GRAPPA" else MRIOperator
    #                                )

    # test_dataloader = MRIDataLoader(test_dataset,
    #                               batch_size=args.batch_size, 
    #                               shuffle=False, 
    #                               operator=GRAPPAOperator if args.operator == "GRAPPA" else MRIOperator
    #                             )
    
    
    
    # Model 
    # model = UNet(in_channels=1, out_channels=1, device=args.device).to(args.device)
    # model = ViT(d_model=512, n_heads=8, patch_size=16, n_channels=1, n_layers=8, dropout=0.2, image_size=(1, 320, 320), device=args.device).to(args.device)
    # model = AttentionUNet(in_channels=1, out_channels=1, device=args.device).to(args.device)
    # model = UNet_SPICER(in_chans=1, out_chans=1).to(args.device)
    model = FCN(in_channels=1, out_channels=1, device=args.device).to(args.device)
    
    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    
    # Set the seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    
    
    # criterion = get_robust_loss()  # Combined MSE + L1 loss
    criterion = get_combined_loss() # Combined MSE + L1 + SSIM + MS SSIM loss
    
    optimizer = get_robust_optimizer(model, initial_lr=args.lr)
    if args.continue_training and args.checkpoint_path:
        print(f"Continuing training from {args.checkpoint_path}")
        continue_training(model, 
                        train_loader=train_dataloader,
                        test_loader=test_dataloader,
                        num_epochs=args.num_epochs,
                        criterion=criterion,
                        optimizer=optimizer,
                        device=args.device,
                        save_path=args.outdir,
                        checkpoint_path=args.checkpoint_path)
    else:
        out_model, best_psnr, best_epoch = improved_train(model,
                                                    train_dataloader, 
                                                    test_dataloader, 
                                                    criterion, 
                                                    optimizer=optimizer, 
                                                    num_epochs=args.num_epochs, 
                                                    device=args.device,
                                                    save_path=args.outdir)


    
    log_file.close()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", parents=[args_parser()])
    args = parser.parse_args()

    main(args)


"""When operator is within DataLoader/Dataset"""
# python main.py --acc 4 --num_coils 8 --acs_lines 32 --kernel_size 3 --operator GRAPPA --batch_size 32 --num_epochs 10 --lr 0.001 --lr2 0.0001 --device cuda:2 --seed 42 --outdir ./Saved/UNet/

"""When operator was done before DataLoader/Dataset (preloaded images)"""
# python main.py --batch_size 32 --num_epochs 50 --lr 1e-4 --device cuda:3 --outdir ./Saved/InstanceNorm/UNet_acc6/ --seed 42

# python main.py --batch_size 32 --num_epochs 50 --lr 1e-2 --device cuda:3 --outdir ./Saved/ViT_Up_3/acc6 --seed 42


# python main.py --batch_size 32 --num_epochs 50 --lr 1e-3 --device cuda:3 --outdir ./Saved/FCN/acc6 --seed 42

"""Continue Training"""
# python main.py --batch_size 16 --num_epochs 150 --lr  1e-2 --device cuda:3 --outdir ./Saved/InstanceNorm/UNet_acc6_higher_lr_cont --seed 42 --continue_training --checkpoint_path ./Saved/InstanceNorm/UNet_acc6_higher_lr/checkpoint_epoch_30.pth