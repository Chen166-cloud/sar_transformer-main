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

from utils import BSD_SAR
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
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', required=True, type=str)
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


train_dataset = BSD_SAR(args.train_dataset, crop_size, training_set=True)
val_dataset = BSD_SAR(args.val_dataset, crop_size, training_set=False)



dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

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

if args.load != "default":
    checkpoint = torch.load(args.load, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # Handle common DataParallel key mismatches between save/load runs.
        if any(k.startswith("module.") for k in checkpoint.keys()):
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        else:
            checkpoint = {f"module.{k}": v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from: {args.load}")


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)
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


def train_model(model, criterion, optimizer, dataloader, valloader, direc, num_epochs=400, record_file='ExperimentalRecord.txt'):
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
    best_psnr = float("nan")
    best_ssim = float("nan")
    early_stop_patience = 20
    min_delta = 1e-6
    no_improve = 0
    best_model_path = os.path.join(direc, "best_model.pth")

    for epoch in range(args.start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        train_epoch_loss = None
        val_epoch_loss = None
        val_epoch_psnr = None
        val_epoch_ssim = None

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                running_loss = 0.0
                running_loss_tv = 0.0
                for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):        
    
                    # X_batch = Variable(X_batch.to(device ='cuda'))
                    # y_batch = Variable(y_batch.to(device='cuda'))
                    X_batch = Variable(X_batch.to(device))
                    y_batch = Variable(y_batch.to(device))

                    output = model(X_batch)

                    # print(output.size())

                
                    loss = criterion(output, y_batch)

                    
                    loss = loss + TV_loss(output,0.0000005)

                    # ===================backward====================
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
                epoch_loss = running_loss / (batch_idx+1)
                train_epoch_loss = epoch_loss
                
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                

                fulldir = direc+ "/all/" +"/{}/".format(epoch)
                
                if not os.path.isdir(fulldir):
                
                    os.makedirs(fulldir)
                torch.save(model.state_dict(), fulldir+args.model+".pth")

            else:
                model.eval()   # Set model to evaluate model
                running_loss = 0.0
                running_psnr = 0.0
                running_ssim = 0.0
                with torch.no_grad():
                    for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):

                        # X_batch = Variable(X_batch.to(device='cuda'))
                        # y_batch = Variable(y_batch.to(device='cuda'))

                        X_batch = Variable(X_batch.to(device))
                        y_batch = Variable(y_batch.to(device))

                        output = model(X_batch)

                        loss = criterion(output, y_batch)

                        running_loss += loss.item()
                        running_psnr += compute_batch_psnr(output, y_batch)
                        running_ssim += compute_batch_ssim(output, y_batch)

                epoch_loss = running_loss / (batch_idx+1)
                val_epoch_loss = epoch_loss
                val_epoch_psnr = running_psnr / (batch_idx + 1)
                val_epoch_ssim = running_ssim / (batch_idx + 1)
                print('{} Loss (MSE): {:.4f}'.format(phase, epoch_loss))
                print('{} PSNR: {:.4f}, SSIM: {:.4f}'.format(phase, val_epoch_psnr, val_epoch_ssim))

                scheduler.step(epoch_loss)

                if epoch_loss < (best_loss - min_delta):
                    best_loss = epoch_loss
                    best_epoch = epoch + 1
                    best_psnr = val_epoch_psnr
                    best_ssim = val_epoch_ssim
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), best_model_path)
                    no_improve = 0
                else:
                    no_improve += 1

        # Append one record line per epoch for downstream experiment tracking.
        with open(record_file, "a") as f:
            f.write(
                "{model}\tepoch={epoch}\tbatch_size={batch_size}\tinitial_lr={initial_lr:.8f}\tcurrent_lr={current_lr:.8f}\tweight_decay={weight_decay:.8f}\tlambda_loss={lambda_loss:.6f}\ttrain_loss={train_loss:.6f}\tval_loss={val_loss:.6f}\tval_psnr={val_psnr:.6f}\tval_ssim={val_ssim:.6f}\n".format(
                    model=args.model,
                    epoch=epoch + 1,
                    batch_size=args.batch_size,
                    initial_lr=args.learning_rate,
                    current_lr=optimizer.param_groups[0]["lr"],
                    weight_decay=optimizer.param_groups[0].get("weight_decay", 0.0),
                    lambda_loss=lambda_loss,
                    train_loss=train_epoch_loss if train_epoch_loss is not None else float("nan"),
                    val_loss=val_epoch_loss if val_epoch_loss is not None else float("nan"),
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
    print('Best val loss: {:4f}'.format(best_loss))
    print(
        "Best model saved at epoch {} | PSNR: {:.4f} | SSIM: {:.4f} | path: {}".format(
            best_epoch, best_psnr, best_ssim, best_model_path
        )
    )

    with open(record_file, "a") as f:
        f.write(
            "{model}\tbest_model\tepoch={epoch}\tbest_val_loss={val_loss:.6f}\tbest_val_psnr={val_psnr:.6f}\tbest_val_ssim={val_ssim:.6f}\tpath={path}\n".format(
                model=args.model,
                epoch=best_epoch,
                val_loss=best_loss,
                val_psnr=best_psnr,
                val_ssim=best_ssim,
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



model_ft = train_model(model, criterion, optimizer, dataloader, valloader, direc, num_epochs, record_file)



                



    