import json
import os
import random

from click import Context
from typer import Typer, Option, Argument
from typer.core import TyperGroup
from rich.progress import Progress, SpinnerColumn, track, TextColumn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path, PosixPath
from tqdm import tqdm
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import PIImageDataset
from loss import MaskedMSELoss, PILoss
from models import UNet, ConvAutoencoder
from utils import set_seed, calc_mask


def get_model(name: str, unet_init_features=32, autoencoder_latent_size=100, input_size=64):
    name = name.lower()

    if name == 'unet':
        return UNet(in_channels=1, out_channels=20, init_features=unet_init_features)

    if name == 'convautoencoder':
        return ConvAutoencoder(
            in_channels=1, out_channels=20,
            image_size=input_size, latent_size=autoencoder_latent_size
        )


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


def build_figure(K, p, p_pred, use_mask=True):
    K = K.detach().cpu().numpy()
    mask = calc_mask(K)
    if not use_mask:
        mask[:] = True
    p = p.detach().cpu().numpy() * mask
    p_pred = p_pred.detach().cpu().numpy() * mask
    f = plt.figure(dpi=150)
    plt.subplot(2, 3, 1)
    plt.imshow(p[0], origin='lower')
    plt.title(r'True $p(x, y, t=1)$')
    plt.subplot(2, 3, 2)
    plt.imshow(p[p.shape[0] // 2], origin='lower')
    plt.title(r'True $p(x, y, t=10)$')
    plt.subplot(2, 3, 3)
    plt.imshow(p[-1], origin='lower')
    plt.title(r'True $p(x, y, t=20)$')
    plt.subplot(2, 3, 4)
    plt.imshow(p_pred[0], origin='lower')
    plt.title(r'Predicted $p(x, y, t=1)$')
    plt.subplot(2, 3, 5)
    plt.imshow(p_pred[p.shape[0] // 2], origin='lower')
    plt.title(r'Predicted $p(x, y, t=10)$')
    plt.subplot(2, 3, 6)
    plt.imshow(p_pred[-1], origin='lower')
    plt.title(r'Predicted $p(x, y, t=20)$')
    plt.tight_layout()
    return f


def compute_loss(mse: nn.Module, pi: nn.Module, alpha: float, p: torch.tensor,
                 p_pred: torch.tensor, K: torch.tensor, use_mask: bool, mask_arg):
    if use_mask:
        if mask_arg != 'precond':
            try:
                mask_arg = int(mask_arg)
            except:
                mask_arg = float(mask_arg)
        mask = calc_mask(K, dilation=mask_arg)
    else:
        mask = None
    if alpha != 1:
        mse = mse(p, p_pred, mask=mask)
    else:
        mse = 0.0
    if alpha == 0:
        return mse
    pi = pi(p_pred, K, mask=mask)

    return mse + alpha * pi


class Model(str, Enum):
    unet = 'UNet'
    convautoencoder = 'ConvAutoencoder'


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)


app = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False)

app_export = Typer(rich_markup_mode='rich', cls=OrderCommands, add_completion=False)


@app.command('train', help='Run training routine...')
def train(file_pressure: Path = Argument(..., help='File containing pressure data in .npy format'),
          file_perm: Path = Argument(..., help='File containing permetivity data in .npy format'),
          run_id: str = Argument(..., help='And ID/name for this particualr run'),
          tmp_folder: Path = Option('results', help='Folder where postprocessed dataset files and tensorboard results shall be stored'),
          model_name: Model = Option(Model.unet.value, help='Model name for a neural network.'),
          alpha_pi: float = Option(0.0, help='Alpha multiplier in front of the PDE loss'),
          n_epoch: int = Option(200, help='Number of training epochs'),
          unet_init_features: int = Option(32, help='Number if channels at the first convoltuinal layer in UNet model'),
          autoencoder_latent_size: int = Option(100, help='Latent space size in Autoencoder model'),
          masking: bool = Option(True, help='Apply masking trick in MSE loss function'),
          mask_dilation: str = Option('3', help='Either integer value specifying number of neighbours for masking technique or'\
                                        ' a string "precond" that will force NN to pay greater attention to cells of higher permetivity'),
          plot_step: int = Option(20, help='Some predictions shall be saved to tensorboard each [cyan]plot_step[/cyan] epochs'),
          train_fraction: float = Option(0.8, help='Fraction of data that will be used for training'),
          batch_size: int = Option(16, help='Batch size'),
          num_workers: int = Option(10, help='Number of workers to be used for dataset preprocessing and for DataLoader'),
          seed: int = Option(42, help='Seed')
              ):
    params = locals()
    print(params)
    for k, v in params.items():
        if type(v) is PosixPath:
            params[k] = str(v)
    model_folder = os.path.join(tmp_folder, run_id)
    os.makedirs(model_folder, exist_ok=True)
    with open(os.path.join(model_folder, 'params.json'), 'w') as f:
        json.dump(params, f)
    set_seed(seed)
    
    model = get_model(
        model_name,
        unet_init_features=unet_init_features,
        autoencoder_latent_size=autoencoder_latent_size,
    ).cuda()
    
    tmp_filename = os.path.split(file_perm)[-1] + '_' + os.path.split(file_pressure)[-1] + '.dataset'
    tmp_filename = os.path.join(tmp_folder, tmp_filename)
    p = Progress(SpinnerColumn(speed=0.5), TextColumn("[progress.description]{task.description}"), transient=True)
    p.add_task(description="Loading data...", total=None)
    p.start()
    dataset = PIImageDataset(file_perm, file_pressure, n_jobs=num_workers, tmp_filename=tmp_filename)
    p.stop()
    train_size = int(train_fraction * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)
    
    perm_mult, press_mult = compute_multipliers(train_dataset)
    
    masked_mse = MaskedMSELoss()
    pi_loss = PILoss()

    optimizer = torch.optim.Adam(model.parameters())
    
    writer = SummaryWriter(log_dir=model_folder, comment=run_id)
    
    for epoch in track(range(n_epoch), description='Training...'):
        model.train(True)
        train_loss = 0.0
        for K, sources, p in dl_train:
            K = K.cuda()
            p = p.cuda()
            with torch.no_grad():
                K = K / perm_mult
                p = p / press_mult
            optimizer.zero_grad()
            p_pred = model(K)
            lf = compute_loss(masked_mse, pi_loss, alpha_pi, p, p_pred, K, masking, mask_dilation)
            lf.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += lf
        writer.add_scalar('Loss/train', train_loss, epoch)
        model.train(False)
        test_loss_masked = 0.0
        test_loss = 0.0
        for K, sources, p in dl_test:
            K = K.cuda() / perm_mult
            p = p.cuda() / press_mult
            with torch.no_grad():
                p_pred = model(K)
                test_loss_masked += masked_mse(p_pred, p, calc_mask(K))
                test_loss += masked_mse(p_pred, p)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Loss/test_masked', test_loss_masked, epoch)
        if not epoch % plot_step:
            with torch.no_grad():
                K, _, p = test_dataset[0]
                p = p / press_mult
                K = K.unsqueeze(0).cuda() / perm_mult
                p_pred = model(K)[0]
                writer.add_figure('train/fig', build_figure(K[0,0], p, p_pred), epoch)
                K, _, p = test_dataset[0]
                p = p / press_mult
                K = K.unsqueeze(0).cuda() / perm_mult
                p_pred = model(K)[0]
                writer.add_figure('test/fig', build_figure(K[0,0], p, p_pred), epoch)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    app()
