from tqdm import tqdm
import numpy as np
import torch
from .PINO_Loss import psi_interpolation, PINO_loss



def eval_gs(model,
            dataloader,
            config,
            device,
            use_tqdm=True):
    model.eval()
    #myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    test_err = []
    pde_err = []
    bcs_err = []

    for xx, yy, xx_inter in pbar:
        xx, yy, xx_inter = xx.to(device), yy.to(device), xx_inter.to(device)

           
        fields_inter = psi_interpolation(xx, yy, xx_inter)   # (batch, 161*361, 101*361, 1)
            
        xx, yy, xx_inter = torch.tensor(xx), torch.tensor(yy), torch.tensor(xx_inter)
        xx, yy, xx_inter, fields_inter = xx.to(device), yy.to(device), xx_inter.to(device), fields_inter.to(device)
        xx_tile = torch.tile(xx, (1, xx_inter.shape[1], xx_inter.shape[1], 1))  # (batch, 161*361, 101*361, 1)
          
        pred = model([fields_inter, ], xx_tile)
            #data_loss = myloss(pred, yy)
        pde_loss, bcs_loss, data_loss = PINO_loss(model, xx_tile, xx, yy, fields_inter, pred, config, xx_inter)
    
        test_err.append(data_loss.item())
        pde_err.append(pde_loss.item())
        bcs_err.append(bcs_loss.item())

    mean_pde_err = np.mean(pde_err)
    std_pde_err = np.std(pde_err, ddof=1) / np.sqrt(len(pde_err))

    mean_bcs_err = np.mean(bcs_err)
    std_bcs_err = np.std(bcs_err, ddof=1) / np.sqrt(len(bcs_err))

    mean_err = np.mean(test_err)
    std_err = np.std(test_err, ddof=1) / np.sqrt(len(test_err))

    print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
          f'==Averaged equation error mean: {mean_pde_err}, std error: {std_pde_err}==\n'
          f'==Averaged Boundary condition error mean: {mean_bcs_err}, std error: {std_bcs_err}==')
