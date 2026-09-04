# Code for training TransSAR on synthetic images 
# Author: Malsha Perera
import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from datetime import datetime




import argparse
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torchvision import transforms as T
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init as init
import math
from utils import RealSARDataset, BSD_SAR
from transform_main import (
    TransSARV2,
    TransSARV2_ModelA,
    TransSARV2_ModelB,
    TransSARV2_Full,
    TransSARV2_Freq,
    TransSARV2_FreqNG,
    TransSARV2_FreqNG_Bottle,
    TransSARV2_DualFreqNG_Bottle,
)




parser = argparse.ArgumentParser(description='TransSAR')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', default=None, type=str)
parser.add_argument('--synthetic_val_dataset', default=None, type=str,
                    help='synthetic validation dataset path for PSNR/SSIM evaluation')
parser.add_argument('--modelname', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', default='TransSARV2', type=str,
                    help='model name')
parser.add_argument('--direc', required=True , type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int ,default=256)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--lambda_loss', default=0.04, type=float)
parser.add_argument('--record_file', default='ExperimentalRecord.txt', type=str,
                    help='file path to append per-epoch training records')
parser.add_argument('--pretrain', type=str, default=None, help='path to pretrained model')
parser.add_argument('--mask_ratio', type=float, default=0.3, help='mask ratio for self-supervised training')
parser.add_argument('--lambda_tv', type=float, default=0.001, help='weight of TV loss')

args = parser.parse_args()

aug = args.aug
direc = args.direc
num_epochs = args.epochs
modelname = args.modelname
crop_size = (args.crop, args.crop)
lambda_loss = args.lambda_loss
record_file = args.record_file

if args.start_epoch < 0 or args.start_epoch >= num_epochs:
    raise ValueError(
        f"--start-epoch must be in [0, {num_epochs - 1}], got {args.start_epoch}"
    )

def total_variation(image_in):

    tv_h = torch.sum(torch.abs(image_in[ :, :-1] - image_in[ :, 1:]))
    tv_w = torch.sum(torch.abs(image_in[ :-1, :] - image_in[ 1:, :]))
    tv_loss = tv_h + tv_w

    return tv_loss 

def TV_loss(im_batch, weight):
    TV_L = 0.0

    for tv_idx in range(len(im_batch)):
        TV_L = TV_L + total_variation(im_batch[tv_idx,0,:,:])

    TV_L = TV_L/len(im_batch)

    return weight*TV_L


def random_mask(x, mask_ratio=0.3):
    """
    x: [B, 1, H, W]
    mask_ratio: masked pixel ratio
    mask = 1 denotes masked region for reconstruction loss
    """
    mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
    mask = (mask < mask_ratio).float()
    masked_x = x * (1.0 - mask)
    return masked_x, mask


def masked_l1_loss(pred, target, mask):
    loss = torch.abs(pred - target) * mask
    return loss.sum() / (mask.sum() + 1e-6)


def tv_loss(x):
    loss_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    loss_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return loss_h + loss_w

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)


def compute_batch_psnr(pred, target, eps=1e-12):
    # Compute sample-wise PSNR with a per-sample dynamic range to avoid NaNs.
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)

    target_flat = target.view(target.size(0), -1)
    data_range = target_flat.max(dim=1).values - target_flat.min(dim=1).values
    data_range = torch.clamp(data_range, min=1e-6)

    psnr = 10.0 * torch.log10((data_range ** 2) / torch.clamp(mse, min=eps))
    return psnr.mean().item()


def compute_batch_ssim(pred, target, data_range=1.0, eps=1e-12):
    # Global SSIM per image (fast approximation for validation trend tracking).
    k1, k2 = 0.01, 0.03
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    mu_x = pred_flat.mean(dim=1)
    mu_y = target_flat.mean(dim=1)

    sigma_x = ((pred_flat - mu_x.unsqueeze(1)) ** 2).mean(dim=1)
    sigma_y = ((target_flat - mu_y.unsqueeze(1)) ** 2).mean(dim=1)
    sigma_xy = ((pred_flat - mu_x.unsqueeze(1)) * (target_flat - mu_y.unsqueeze(1))).mean(dim=1)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    ssim = numerator / torch.clamp(denominator, min=eps)
    return ssim.mean().item()


train_dataset = RealSARDataset(args.train_dataset)
val_dataset = RealSARDataset(args.val_dataset) if args.val_dataset is not None else None
synthetic_val_dataset = BSD_SAR(
    args.synthetic_val_dataset,
    crop_size,
    training_set=False
) if args.synthetic_val_dataset is not None else None



dataloader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers if hasattr(args, 'workers') else 0
)
valloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=args.workers if hasattr(args, 'workers') else 0
) if val_dataset is not None else None
synthetic_valloader = DataLoader(
    synthetic_val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.workers if hasattr(args, 'workers') else 0
) if synthetic_val_dataset is not None else None

syn_debug_loader = DataLoader(
    BSD_SAR(args.synthetic_val_dataset, crop_size, training_set=False),
    batch_size=1,
    shuffle=False
)

real_debug_loader = DataLoader(
    RealSARDataset(args.train_dataset),
    batch_size=1,
    shuffle=False
)

syn_x, syn_y, *rest = next(iter(syn_debug_loader))
real_x = next(iter(real_debug_loader))

if isinstance(real_x, dict):
    real_x = real_x["noisy"]
elif isinstance(real_x, (list, tuple)):
    real_x = real_x[0]

print("synthetic noisy:",
      syn_x.shape,
      syn_x.min().item(),
      syn_x.max().item(),
      syn_x.mean().item())

print("synthetic clean:",
      syn_y.shape,
      syn_y.min().item(),
      syn_y.max().item(),
      syn_y.mean().item())

print("real noisy:",
      real_x.shape,
      real_x.min().item(),
      real_x.max().item(),
      real_x.mean().item())

# device = torch.device("cuda")
#
#
# model = TransSARV2()
#
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   model = nn.DataParallel(model,device_ids=[0,1]).cuda()
# model.to(device)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if args.model == "TransSARV2":
    model = TransSARV2()
elif args.model == "TransSARV2_ModelA":
    model = TransSARV2_ModelA()
elif args.model == "TransSARV2_ModelB":
    model = TransSARV2_ModelB()
elif args.model == "TransSARV2_Full":
    model = TransSARV2_Full()
elif args.model == "TransSARV2_Freq":
    model = TransSARV2_Freq()
elif args.model == "TransSARV2_FreqNG":
    model = TransSARV2_FreqNG()
elif args.model == "TransSARV2_FreqNG_Bottle":
    model = TransSARV2_FreqNG_Bottle()
elif args.model == "TransSARV2_DualFreqNG_Bottle":
    model = TransSARV2_DualFreqNG_Bottle()
else:
    raise ValueError(f"Unsupported model: {args.model}")

if torch.cuda.device_count() > 1 and device.type == "cuda":
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=[0, 1])

model = model.to(device)

ckpt_path = None

if args.pretrain is not None:
    ckpt_path = args.pretrain
elif args.load != "default":
    ckpt_path = args.load

if ckpt_path is not None:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # Handle common DataParallel key mismatches between save/load runs.
        if any(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        else:
            checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from: {ckpt_path}")


# ================= Full-model fine-tuning: train all parameters =================
for _, param in model.named_parameters():
    param.requires_grad = True

trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable params (full fine-tune):", trainable_count)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.learning_rate,
    weight_decay=1e-5
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=True
)




pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


def train_model(model, criterion, optimizer, dataloader, valloader, synthetic_valloader, direc, num_epochs=400, record_file='ExperimentalRecord.txt'):
    os.makedirs(direc, exist_ok=True)
    since = time.time()
    train_start_time = datetime.now()

    with open(record_file, "a") as f:
        f.write(
            "{model}\ttrain_start\tstart_time={start_time}\tstart_epoch={start_epoch}\tload={load}\n".format(
                model=args.model,
                start_time=train_start_time.strftime("%Y-%m-%d %H:%M:%S"),
                start_epoch=args.start_epoch,
                load=args.load,
            )
        )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = -1
    early_stop_patience = 20
    min_delta = 1e-6
    no_improve = 0
    best_model_path = os.path.join(direc, "best_model.pth")

    # ===== Evaluate synthetic val before self-supervised fine-tuning =====
    if synthetic_valloader is not None:
        model.eval()
        synthetic_loss_total = 0.0
        synthetic_psnr_total = 0.0
        synthetic_ssim_total = 0.0
        synthetic_count = 0

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch, *rest) in enumerate(synthetic_valloader):
                noisy = Variable(X_batch.to(device).float())
                clean = Variable(y_batch.to(device).float())

                output = model(noisy)

                syn_loss = F.mse_loss(output, clean)

                synthetic_loss_total += syn_loss.item()
                synthetic_psnr_total += compute_batch_psnr(output, clean)
                synthetic_ssim_total += compute_batch_ssim(output, clean)
                synthetic_count += 1

        pre_syn_loss = synthetic_loss_total / max(synthetic_count, 1)
        pre_syn_psnr = synthetic_psnr_total / max(synthetic_count, 1)
        pre_syn_ssim = synthetic_ssim_total / max(synthetic_count, 1)

        print(
            "[Before selfsup] synthetic_val_loss={:.6f} | synthetic_val_psnr={:.6f} | synthetic_val_ssim={:.6f}".format(
                pre_syn_loss,
                pre_syn_psnr,
                pre_syn_ssim
            )
        )

    for epoch in range(args.start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_epoch_loss = None
        val_epoch_loss = None
        val_epoch_psnr = None
        val_epoch_ssim = None
        val_epoch_selfsup_loss = None

        # For self-supervised training, validation phase is optional.
        phases = ['train', 'val'] if valloader is not None else ['train']
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_loss_tv = 0.0
                for batch_idx, noisy in enumerate(dataloader):
                    noisy = Variable(noisy.to(device).float())
                    masked_noisy, mask = random_mask(noisy, mask_ratio=args.mask_ratio)
                    output = model(masked_noisy)

                    loss_rec = masked_l1_loss(output, noisy, mask)
                    loss_tv = tv_loss(output)
                    loss = loss_rec + args.lambda_tv * loss_tv

                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    running_loss_tv += loss_tv.item()

                epoch_loss = running_loss / (batch_idx+1)
                train_epoch_loss = epoch_loss
                
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                if valloader is None:
                    scheduler.step(epoch_loss)
                    if epoch_loss < (best_loss - min_delta):
                        best_loss = epoch_loss
                        best_epoch = epoch + 1
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), best_model_path)
                        no_improve = 0
                    else:
                        no_improve += 1
                

                fulldir = direc+ "/all/" +"/{}/".format(epoch)
                
                if not os.path.isdir(fulldir):
                
                    os.makedirs(fulldir)
                torch.save(model.state_dict(), fulldir+args.model+".pth")

            else:
                model.eval()   # Set model to evaluate model
                running_loss = 0.0
                with torch.no_grad():
                    for batch_idx, val_data in enumerate(valloader):
                        if isinstance(val_data, dict):
                            noisy = val_data['noisy']
                        elif isinstance(val_data, (list, tuple)):
                            noisy = val_data[0]
                        else:
                            noisy = val_data

                        noisy = Variable(noisy.to(device).float())
                        masked_noisy, mask = random_mask(noisy, mask_ratio=args.mask_ratio)
                        output = model(masked_noisy)

                        loss_rec = masked_l1_loss(output, noisy, mask)
                        loss_smooth = tv_loss(output)
                        loss = loss_rec + args.lambda_tv * loss_smooth

                        running_loss += loss.item()

                epoch_loss = running_loss / (batch_idx+1)
                val_epoch_loss = epoch_loss
                val_epoch_selfsup_loss = epoch_loss
                print('val_selfsup_loss={:.6f}'.format(epoch_loss))

                scheduler.step(epoch_loss)

                if epoch_loss < (best_loss - min_delta):
                    best_loss = epoch_loss
                    best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_model_path)
                    no_improve = 0
                else:
                    no_improve += 1

        # Synthetic validation: compute PSNR / SSIM on paired synthetic SAR val set.
        if synthetic_valloader is not None:
            model.eval()
            synthetic_loss_total = 0.0
            synthetic_psnr_total = 0.0
            synthetic_ssim_total = 0.0
            synthetic_count = 0

            with torch.no_grad():
                for batch_idx, (X_batch, y_batch, *rest) in enumerate(synthetic_valloader):
                    noisy = Variable(X_batch.to(device).float())
                    clean = Variable(y_batch.to(device).float())

                    output = model(noisy)

                    syn_loss = F.mse_loss(output, clean)

                    synthetic_loss_total += syn_loss.item()
                    synthetic_psnr_total += compute_batch_psnr(output, clean)
                    synthetic_ssim_total += compute_batch_ssim(output, clean)
                    synthetic_count += 1

            synthetic_val_loss = synthetic_loss_total / max(synthetic_count, 1)
            val_epoch_psnr = synthetic_psnr_total / max(synthetic_count, 1)
            val_epoch_ssim = synthetic_ssim_total / max(synthetic_count, 1)

            print(
                "synthetic_val_loss={:.6f} | synthetic_val_psnr={:.6f} | synthetic_val_ssim={:.6f}".format(
                    synthetic_val_loss,
                    val_epoch_psnr,
                    val_epoch_ssim
                )
            )

        # Append one record line per epoch for downstream experiment tracking.
        with open(record_file, "a") as f:
            f.write(
                "{model}\tepoch={epoch}\tbatch_size={batch_size}\tinitial_lr={initial_lr:.8f}\tcurrent_lr={current_lr:.8f}\tweight_decay={weight_decay:.8f}\tlambda_loss={lambda_loss:.6f}\ttrain_loss={train_loss:.6f}\tval_loss={val_loss:.6f}\tval_selfsup_loss={val_selfsup_loss:.6f}\tval_psnr={val_psnr:.6f}\tval_ssim={val_ssim:.6f}\n".format(
                    model=args.model,
                    epoch=epoch + 1,
                    batch_size=args.batch_size,
                    initial_lr=args.learning_rate,
                    current_lr=optimizer.param_groups[0]["lr"],
                    weight_decay=optimizer.param_groups[0].get("weight_decay", 0.0),
                    lambda_loss=lambda_loss,
                    train_loss=train_epoch_loss if train_epoch_loss is not None else float("nan"),
                    val_loss=val_epoch_loss if val_epoch_loss is not None else float("nan"),
                    val_selfsup_loss=val_epoch_selfsup_loss if val_epoch_selfsup_loss is not None else float("nan"),
                    val_psnr=val_epoch_psnr if val_epoch_psnr is not None else float("nan"),
                    val_ssim=val_epoch_ssim if val_epoch_ssim is not None else float("nan"),
                )
            )

        if no_improve >= early_stop_patience:
            print("Early stopping.")
            break


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val_selfsup_loss: {:4f}'.format(best_loss))
    print(
        "Best model saved at epoch {} | path: {}".format(best_epoch, best_model_path)
    )

    with open(record_file, "a") as f:
        f.write(
            "{model}\tbest_model\tepoch={epoch}\tbest_val_selfsup_loss={val_loss:.6f}\tpath={path}\n".format(
                model=args.model,
                epoch=best_epoch,
                val_loss=best_loss,
                path=best_model_path,
            )
        )
        f.write(
            "{model}\ttrain_end\tend_time={end_time}\tduration_sec={duration:.2f}\n".format(
                model=args.model,
                end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                duration=time_elapsed,
            )
        )


    model.load_state_dict(best_model_wts)
    return model



model_ft = train_model(
    model,
    criterion,
    optimizer,
    dataloader,
    valloader,
    synthetic_valloader,
    direc,
    num_epochs,
    record_file
)



                



    