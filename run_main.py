from argparse import ArgumentParser
import yaml
import torch
from src.model import DeepONetMulti
from src.dataloader import GSLoader
from src.train_2d_gs import train_gs
from src.eval_2d_gs import eval_gs


def run(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = GSLoader(data_config['datapath'])
    train_loader = dataset.make_loader(n_sample=data_config['n_sample'],
                                       batch_size=config['train']['batchsize'],
                                       start=data_config['offset'], train=True)

    # model = DeepONet(branch_layer=[u0_dim] + config['model']['branch_layers'],
    #                  trunk_layer=[2] + config['model']['trunk_layers']).to(device)
    
    model = DeepONetMulti(input_dim=data_config['coords_shape'], 
                          operator_dims=[data_config['intep_psi_shape'], ], 
                          output_dim=data_config['fields_shape'], 
                          planes_branch=config['model']['branch_layers'], 
                          planes_trunk=config['model']['trunk_layers']).to(device)
    
    # Load from checkpoint
    if 'ckpt' in config['train']:
        ckpt_path = config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    optimizer = torch.optim.Adam(model.parameters(), betas=config['train']['betas'],
                     lr=config['train']['base_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['train']['milestones'],
                                                     gamma=config['train']['scheduler_gamma'])
    train_gs(model,
            train_loader,
            optimizer,
            scheduler,
            config,
            device)


def test(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    dataset = GSLoader(data_config['datapath'])
    dataloader = dataset.make_loader(n_sample=data_config['n_sample'],
                                     batch_size=config['test']['batchsize'],
                                     start=data_config['offset'])

    model = DeepONetMulti(input_dim=data_config['coords_shape'], 
                          operator_dims=[data_config['intep_psi_shape'], ], 
                          output_dim=data_config['fields_shape'], 
                          planes_branch=config['model']['branch_layers'], 
                          planes_trunk=config['model']['trunk_layers']).to(device)
    
    if 'ckpt' in config['test']:
        ckpt_path = config['test']['ckpt']
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)

    eval_gs(model, dataloader, config, device)


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--mode', type=str, help='train or test')
    args = parser.parse_args()

    config_file = args.config_path = 'config/train/gs.yaml'
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    # if args.mode == 'train':
    #     run(config)
    # else:
    #     test(config)
    run(config)
