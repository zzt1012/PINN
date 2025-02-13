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

        fields_inter = psi_interpolation(xx, yy, xx_inter)
        xx = torch.tile(fields_inter, (1, fields_inter.shape[1], fields_inter.shape[2], 1))  #  (batch, 161*361, 101*361, 1)
        
        out = model([fields_inter, ], xx)
        #data_loss = myloss(out, yy)

        bcs_loss, pde_loss, data_loss = PINO_loss(xx, yy, fields_inter, out, config, xx_inter)
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