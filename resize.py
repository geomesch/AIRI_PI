from click import Context
from typer import Typer, Option, Argument
from typer.core import TyperGroup
from rich.progress import Progress, SpinnerColumn, track, TextColumn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, InterpolationMode
from dataset import PIImageDataset
from models import UpconvNet
from utils import set_seed, calc_mask
from torch import nn
from loss import MaskedMSELoss, PILoss
from pathlib import Path, PosixPath
from tqdm import tqdm
from enum import Enum
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gf
import numpy as np
import random
import json
import glob
import torch
import os


def compute_multipliers(train_dataset):
    # t, n, m = train_dataset[0][-1].shape
    perm_mult = 1.0
    press_mult = 1.0
    # press_mult = torch.ones(t)
    for perm, _, pressure in train_dataset:
        # mask = perm != 0
        perm_mult = max(perm_mult, perm.max())
        press_mult = max(press_mult, pressure.max())
        # mx = (pressure * mask).reshape(t, -1).max(axis=1)
    return perm_mult, press_mult

file_perm = 'data/perm1_172.npy'
file_pressure = 'data/pressure1_172.npy'

file_low_perm = 'data/perm1_470.npy'
file_low_pressure = 'data/pressure1_470.npy'

tmp_folder = 'results'
num_workers = 10
train_fraction = 0.95
batch_size = 8
n_epoch = 150
seed = 1
set_seed(seed)


p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
p.add_task(description="Loading data...", total=None)
p.start()
tmp_filename = os.path.split(file_perm)[-1] + '_' + os.path.split(file_pressure)[-1] + '.dataset'
tmp_filename = os.path.join(tmp_folder, tmp_filename)
dataset = PIImageDataset(file_perm, file_pressure, n_jobs=num_workers, tmp_filename=tmp_filename)
tmp_filename = os.path.split(file_low_perm)[-1] + '_' + os.path.split(file_low_pressure)[-1] + '.dataset'
tmp_filename = os.path.join(tmp_folder, tmp_filename)
dataset_low = PIImageDataset(file_low_perm, file_low_pressure, n_jobs=num_workers, tmp_filename=tmp_filename)
p.stop()
train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
perm_mult, press_mult = compute_multipliers(train_dataset)

dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dl_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)
dl_low = DataLoader(dataset_low, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
R_target = Resize([128, 128], interpolation=InterpolationMode.NEAREST_EXACT).cuda()
R_given = Resize([64, 64], interpolation=InterpolationMode.NEAREST_EXACT).cuda()
loss = MaskedMSELoss().cuda()
# Pressures

def extract_data(t, pressure: bool, time: int, perm_mult, press_mult):
    with torch.no_grad():
        K, sources, p = t
        K = K / perm_mult
        p = p / press_mult
        p[(K== 0).repeat(1, 20, 1, 1)] = 0.0
        if not pressure:
            z = K
        else:
            z = p[:, time:time + 1]
        x = R_given(z)
        x[x == 0] = -1
        y = R_target(z)
        y[y == 0] = -1
    return x.cuda(), y.cuda()

def postprocess_output(x):
    x = torch.clip(x, 0.0, 1.0).detach().cpu()
    res = list()
    for item in x:
        item = gf(item.squeeze().numpy(), sigma=0.5)
        res.append(item)
    return np.array(res)
        
    
pressure = False
time = 0
lowscale_dataset = dl_low
def upscale(pressure: bool, time: int, perm_mult: float, press_mult: float, lowscale_dataset):
    run_id = 'upscale_pressure' + str(time) if pressure else 'upscale_perm'
    model_folder = os.path.join(tmp_folder, run_id)
    model = UpconvNet(scaling_factor=2).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    writer = SummaryWriter(log_dir=model_folder, comment=run_id)
    print(run_id)
    files = glob.glob(model_folder + '/*')
    for f in files:
        os.remove(f)
    best_state = None
    best_loss = float('inf')
    for epoch in track(range(n_epoch), description='Training...'):
        model.train(True)
        train_loss = 0.0
        for t in dl_train:
            x, y = extract_data(t, pressure, time, perm_mult, press_mult)
            optimizer.zero_grad()
            y_pred = model(x)
            lf = loss(y_pred, y) #+ 0.2 * loss(nn.functional.interpolate(y_pred, scale_factor=0.5, mode='nearest-exact'), x)
            lf.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += lf
        writer.add_scalar('Loss/train', train_loss, epoch)
        model.train(False)
        test_loss = 0.0
        for t in dl_test:
            x, y = extract_data(t, pressure, time, perm_mult, press_mult)
            with torch.no_grad():
                y_pred = model(x)
                y_pred = torch.clip(y_pred, 0.0)
                y = torch.clip(y, 0.0)
                test_loss += loss(y_pred, y)
                TY, TYPRED = y, y_pred
        writer.add_scalar('Loss/test', test_loss, epoch)
        if test_loss < best_loss:
            best_state = model.state_dict()
            best_loss = test_loss
        if not epoch % 10:
            fig = plt.figure(dpi=200)
            y_pred = y_pred[0][0].detach().cpu().numpy()
            y = y[0][0].detach().cpu().numpy()
            y_interp = Resize([128, 128], interpolation=InterpolationMode.NEAREST_EXACT)(x)[0][0].detach().cpu()
            y_filter = gf(y_pred, sigma=0.5) * (y != 0)
            plt.subplot(1, 4, 1)
            plt.imshow(y, origin='lower')
            plt.title('Ground truth')
            plt.subplot(1, 4, 2)
            plt.imshow(y_interp, origin='lower')
            plt.title('[transform.Resize]')
            plt.yticks([])
            plt.subplot(1, 4, 3)
            plt.imshow(y_pred, origin='lower')
            plt.title('Prediction')
            plt.yticks([])
            plt.subplot(1, 4, 4)
            plt.title('Prediction (filtered)')
            plt.imshow(y_filter, origin='lower')
            plt.yticks([])
            plt.tight_layout()
            writer.add_figure('test/result', fig, epoch)
    writer.flush()
    writer.close()
    torch.save(model.state_dict(), os.path.join(model_folder, 'model.pt'))
            # plt.savefig(os.path.join(model_folder, 'example.pdf'))
    model.load_state_dict(best_state)
    plt.figure(dpi=200)
    y_pred = y_pred[0][0].detach().cpu().numpy()
    y = y[0][0].detach().cpu().numpy()
    y_interp = Resize([128, 128])(x)[0][0].detach().cpu()
    y_filter = gf(y_pred, sigma=0.5) * (y != 0)
    plt.subplot(1, 4, 1)
    plt.imshow(y, origin='lower')
    plt.title('Ground truth')
    plt.subplot(1, 4, 2)
    plt.imshow(y_interp, origin='lower')
    plt.title('[transform.Resize]')
    plt.yticks([])
    plt.subplot(1, 4, 3)
    plt.imshow(y_pred, origin='lower')
    plt.title('Prediction')
    plt.yticks([])
    plt.subplot(1, 4, 4)
    plt.title('Prediction (filtered)')
    plt.imshow(y_filter, origin='lower')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, 'example.pdf'))
    
    res = None
    for t in lowscale_dataset:
        x, _ = extract_data(t, pressure, time, perm_mult, press_mult)
        y = model(x)
        y = postprocess_output(y)
        if res is None:
            res = y
        else:
            res = np.append(res, y, axis=0)
    m = press_mult if pressure else perm_mult
    with open(os.path.join(model_folder, 'res.npy'), 'wb') as f:
        np.save(f, res * m.detach().cpu().numpy())
    return res, TY, TYPRED

# upscale(False, 0, perm_mult, press_mult, dl_low)
for i in range(19, 20):
    r, a, b = upscale(True, i, perm_mult, press_mult, dl_low)

res = []
for i in range(0, 20):
    run_id = 'upscale_pressure' + str(i)
    filename = os.path.join(tmp_folder, run_id, 'res.npy')
    with open(filename, 'rb') as f:
        r = np.load(f)
        res.append(r[:, np.newaxis, ...])
res = np.concatenate([np.zeros_like(res[0])] + res, axis=1)
with open('data/pressure_sr.npy', 'wb') as f:
    np.save(f, res)