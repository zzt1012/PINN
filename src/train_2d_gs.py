import numpy as np
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from .PINO_Loss import PINO_loss, psi_interpolation


def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)

def train_gs(model,
            train_loader, 
            optimizer, 
            scheduler,
            config,
            device,
            use_tqdm=True):

    pde_weight = config['train']['pde_loss']
    bcs_weight = config['train']['bcs_loss']
    data_weight = config['train']['data_loss']
    model.train()
    #myloss = LpLoss(size_average=True)
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    for i in pbar:
        model.train()
        train_data = 0.0
        train_pde = 0.0
        train_bcs = 0.0
        train_loss = 0.0

        for xx, yy, xx_inter in train_loader:
            xx, yy, xx_inter = xx.cpu().numpy(), yy.cpu().numpy(), xx_inter.cpu().numpy()
            # print('yy.shape:', yy.shape)
    
            fields_inter = psi_interpolation(xx, yy, xx_inter)   # (batch, 161*361, 101*361, 1)
            
            xx, yy, xx_inter = torch.tensor(xx), torch.tensor(yy), torch.tensor(xx_inter)
            xx, yy, xx_inter, fields_inter = xx.to(device), yy.to(device), xx_inter.to(device), fields_inter.to(device)
            xx_tile = torch.tile(xx, (1, xx_inter.shape[1], xx_inter.shape[1], 1))  # (batch, 161*361, 101*361, 1)
          
            pred = model([fields_inter, ], xx_tile)
            #data_loss = myloss(pred, yy)

            pde_loss, bcs_loss, data_loss = PINO_loss(model, xx_tile, xx, yy, fields_inter, pred, config, xx_inter)
            total_loss = pde_loss * pde_weight + bcs_loss * bcs_weight + data_loss * data_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_data += data_loss.item()
            train_pde += pde_loss.item()
            train_bcs += bcs_loss.item()
            train_loss += total_loss.item()
        scheduler.step()

        train_data /= len(train_loader)
        train_pde /= len(train_loader)
        train_bcs /= len(train_loader)
        train_loss /= len(train_loader)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {i+1}, train loss: {train_loss:.5f}:'
                    f'train data error: {train_data:.5f}; '
                    f'train pde error: {train_pde:.5f};'
                    f'train bcs error: {train_bcs:.5f}'
                )
            )

        if i % 100 == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{i}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')