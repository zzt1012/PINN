import torch
import torch.nn as nn
import time
from torch.autograd import grad
import numpy as np
from src.dataloader import DataNormer
from scipy.interpolate import griddata


def Loss():

    L2loss = nn.MSELoss().cuda()

    return L2loss

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, 
                                   x, 
                                   grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)
    
def psi_interpolation(inn_var, out_var, xx_inter):

    batchsize = out_var.shape[0]
    coords_Rin, coords_Zin = xx_inter[:, :, (0,)], xx_inter[:, :, (1,)]

    PSI_interpolatored = np.zeros((batchsize, coords_Rin.shape[1], coords_Rin.shape[1]))

    for i in range(batchsize):
        coords_ = inn_var[i, :, :, :].reshape(-1, 2)
        field_ = out_var[i, :, :, :].reshape(-1, 1)

        coords_Rin_sample = coords_Rin[i, :, :].squeeze()  # (361,)
        coords_Zin_sample = coords_Zin[i, :, :].squeeze()  # (361,)
        coords_Rin_grid, coords_Zin_grid = np.meshgrid(coords_Rin_sample, coords_Zin_sample, indexing='ij')

        PSI_interpolator = griddata((coords_[:, 0], coords_[:, 1]), field_.squeeze(),
                                (coords_Rin_grid, coords_Zin_grid), method='cubic')     # (361, 361)

        PSI_interpolatored[i] = PSI_interpolator  #(batch, 361, 361)

    #inter_field = PSI_interpolatored.unsqueeze(-1)
    inter_field = np.expand_dims(PSI_interpolatored, axis=-1)
    inter_field = torch.tile(torch.tensor(inter_field, dtype=torch.float32), (1, out_var.shape[1], out_var.shape[2], 1))    # (batch, 161*361, 101*361, 1)

    #print('inter_field:', inter_field.shape)
    
    return  inter_field

def PINO_loss(model, inn_var, xx, true_var, fields_inter, out_var, config, xx_inter):

    R, Z = inn_var[..., (0,)], inn_var[..., (1,)]
    psi = DataNormer(out_var.detach().cpu().numpy(), method='min-max', axis=(0,))
    psi_norm = torch.tensor(psi.norm(out_var), dtype=float)

    inn_var.requires_grad_(True)
    out_var = model([fields_inter, ], inn_var)
   
    dpsida = gradients(out_var, inn_var)
    #print(dpsida.shape)
    dpsidr, dpsidz = dpsida[..., (0,)], dpsida[..., (1,)]
    d2psidr2 = gradients(dpsidr.sum(), inn_var)[:, (0,)]
    d2psidz2 = gradients(dpsidz.sum(), inn_var)[:, (1,)]

    dPdpsi_norm = - config['parameters']['beta0'] * config['parameters']['lambda'] / config['parameters']['R0'] * (1 - psi_norm ** config['parameters']['np']) ** config['parameters']['mp']
    dF2dpsi_norm = -2 * config['parameters']['lambda'] * config['parameters']['miu0'] * config['parameters']['R0'] * (1 - config['parameters']['beta0']) * (1 - psi_norm ** config['parameters']['nf']) ** config['parameters']['mf']
    res_i = d2psidr2 - dpsidr / (R + 1e-8)  + d2psidz2 + config['parameters']['miu0'] * (R + 1e-8) ** 2 * dPdpsi_norm + (1 / 2) * dF2dpsi_norm

    pred_psi_inter = torch.tensor(psi_interpolation(xx.detach().cpu().numpy(), 
                                      out_var[:, :xx.shape[1], :xx.shape[2], :].detach().cpu().numpy(), 
                                      xx_inter.detach().cpu().numpy()), dtype=float)
    # 内部曲线的 psi
    Loss = nn.MSELoss()
    bcs_loss = Loss(pred_psi_inter.cuda(), fields_inter)
    pde_loss = Loss(res_i.float(), torch.zeros_like(res_i, dtype=torch.float32).cuda())
    data_loss = Loss(out_var[:, :xx.shape[1], :xx.shape[2], :], true_var)

    return pde_loss, bcs_loss, data_loss
