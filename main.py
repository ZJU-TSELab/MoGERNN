import torch
from torch import nn
import pytorch_lightning as pl
from data_provider.data_factory import data_provider
from args import initialize_args
from utils.logtool import print_config, print_header, print_train, set_wandb
from experiments import train_model, inference_model
from utils.evaluation import SlitPhyLoss, PinBallLoss, rmse_loss, MSEPinballLoss, MultiPinBallLoss, masked_gaussian_nll_loss
from omegaconf import OmegaConf
from model.regist_model import model_dict
from utils.report_results import report_error, load_prediction
import numpy as np
if __name__ == "__main__":
    args = initialize_args()
    args.exp_id = f'{args.input_len}-{args.pred_len}-{args.look_back}-{args.slide_step}-{args.input_features}-{args.target_features}-t_mark-{args.t_mark}-pos_mark-{args.pos_mark}-seed-{args.seed}'
    if args.unknown_nodes_path:
        per = args.unknown_nodes_path.split('/')[-1].split('_')[-1].split(
            '.')[0]
        if '%' in per:
            args.exp_id = f'{args.dloader_name}-{per}-{args.exp_id}'
        else:
            args.exp_id = f'{args.dloader_name}-{args.exp_id}'

    if args.model == 'woMoERNN':
        if args.mean_expert and args.weight_expert and args.max_expert and args.min_expert and args.diffusion_expert:
            model_wo = 'avg'
        elif args.mean_expert:
            model_wo = 'mean'
        elif args.weight_expert:
            model_wo = 'weight'
        elif args.max_expert:
            model_wo = 'max'
        elif args.min_expert:
            model_wo = 'min'
        elif args.diffusion_expert:
            model_wo = 'diffusion'
        else:
            raise NotImplementedError
        args.exp_id = f'{args.model}-{model_wo}-{args.exp_id}'
    else:
        args.exp_id = f'{args.model}-{args.exp_id}'
    wandb = set_wandb(args)
    configs = OmegaConf.create(vars(args))
    pl.seed_everything(args.seed)
    slide_step = args.slide_step
    args.slide_step = 1  # use 1 for train to collect more data
    train_loader = data_provider(args, 'train')
    args.slide_step = slide_step
    val_loader = data_provider(args, 'val')
    test_loader = data_provider(args, 'test')
    data_loaders = dict(
        zip(['train', 'val', 'test'], [train_loader, val_loader, test_loader]))
    loss_candidate = dict(
        zip([
            'mse', 'huber', 'mae', 'rmse', 'msepinball', 'multipinball', 'nll'
        ], [
            nn.MSELoss(),
            nn.HuberLoss(),
            nn.L1Loss(),
            MSEPinballLoss(),
            MultiPinBallLoss(), masked_gaussian_nll_loss
        ]))
    loss = loss_candidate[args.loss]
    criterions = dict(
        zip([args.loss, 'split_phy_loss'],
            [loss, SlitPhyLoss(features=args.target_features)]))
    model = model_dict[args.model].Model(args).float()
    if args.test_for_changed or args.inference_only:
        args.best_checkpoint_path = args.best_checkpoint_path + args.exp_id + '/'
        if args.pretrained_model_path is None:
            try:
                model_dict = torch.load(
                    args.best_checkpoint_path +
                    'best_val_checkpoint.pth')['state_dict']
                print(f'load pretrained from {args.best_checkpoint_path}')
            except FileNotFoundError:
                model_dict = torch.load(
                    args.best_checkpoint_path +
                    'best_train_checkpoint.pth')['state_dict']
                print(f'load pretrained from {args.best_checkpoint_path}')
            model.load_state_dict(model_dict)
        else:
            model = torch.load(args.pretrained_model_path)
        model.to(args.device).float()
        args.slide_step = 12
        test_loader = data_provider(args, 'test')
        inference_model(model, test_loader, args, criterions)
        exit()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    args.num_params = sum([np.prod(p.size()) for p in model_parameters])
    optimizers = {
        'adamw': torch.optim.AdamW,
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }
    optimizer = optimizers[args.optimizer](model.parameters(),
                                           lr=args.lr,
                                           weight_decay=args.weight_decay)
    schdulers = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'step': torch.optim.lr_scheduler.StepLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
    }
    if args.scheduler == 'plateau':
        scheduler = schdulers[args.scheduler](optimizer,
                                              mode='min',
                                              factor=0.5,
                                              min_lr=1e-5,
                                              patience=args.scheduler_patience)
    elif args.scheduler == 'cosine':
        scheduler = schdulers[args.scheduler](optimizer, T_max=5, eta_min=1e-7)
    elif args.scheduler == 'step':
        scheduler = schdulers[args.scheduler](
            optimizer,
            step_size=args.scheduler_patience,
            gamma=0.5,
        )
    else:

        class NullScheduler:

            def step(self, loss):
                pass

        scheduler = NullScheduler()
    if args.verbose:
        print_header('*** Model Configurations ***')
        print(model)
    print_header('*** Experiment Configurations *** ')
    print_config(configs)
    print_train(args)
    print(f'├── train data dim: {len(train_loader.dataset)}')  #type:ignore
    print(f'├── val   data dim: {len(val_loader.dataset)}')  #type:ignore
    print(f'└── test  data dim: {len(test_loader.dataset)}')  #type:ignore
    pretrained = train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loaders=data_loaders,
        criterions=criterions,
        args=args,
        wandb=wandb,
    )
    if args.save_model:
        import os
        if not os.path.exists('./pretrained'):
            os.makedirs('./pretrained')
        torch.save(pretrained, f'./pretrained/{args.exp_id}.pth')

    path = f'{args.best_prediction_path}best_total_y.pkl'
    y_pred, y, AAS, VS, FS, NAS = load_prediction(path, 'val')
    report_error(y_pred, y, AAS, VS, FS, NAS,
                 f'{args.model}-{args.dloader_name}', './reports.csv')
    print("-> Done!")
