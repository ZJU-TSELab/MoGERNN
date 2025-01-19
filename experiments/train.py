"""
Training functions and helpers
"""
import torch
import pickle
import torch
import os
import time
from .test import read_out

from .inductive_epoch import shared_step


def run_epoch(model,
              data_loaders,
              optimizer,
              scheduler,
              criterions,
              args,
              epoch,
              train=True):
    metrics = {}
    total_y = {mode: {} for mode in data_loaders.keys()}

    for mode, dataloader in data_loaders.items():
        if 'train' in args.stop_based and mode == 'val':
            continue
        # you can define your own shared_step function here to set the desired training process
        st = time.time()
        model, mode_metrics, total_y_mode = shared_step(
            model, dataloader, optimizer, criterions, args, mode, train, epoch)
        total_time = time.time() - st
        print(f'├── {mode} finished, took {total_time:.2f}s')
        if mode == 'test':
            metrics.update({
                'test_always_known_nodes': mode_metrics[0],
                'test_never_known_nodes ': mode_metrics[1],
                'test_new_add_nodes     ': mode_metrics[2],
                'test_failed_nodes      ': mode_metrics[3]
            })
        else:
            metrics.update({
                f'{mode}_total ': mode_metrics[0],
                f'{mode}_mask  ': mode_metrics[1],
                f'{mode}_unmask': mode_metrics[2],
            })
        total_y[mode] = total_y_mode

    if train:
        if 'val' in args.stop_based:
            metric_mode = 'val'
            if args.stop_based == 'val_total':
                metric_name = 'val_total '
            else:
                metric_name = 'val_mask  '
        elif 'train' in args.stop_based:
            metric_mode = 'train'
            if args.stop_based == 'train_total':
                metric_name = 'train_total '
            else:
                metric_name = 'train_mask  '
        save_checkpoint(model, optimizer, args, epoch, metric_mode,
                        metrics[metric_name][args.loss], total_y, metric_name)
        save_checkpoint(model, optimizer, args, epoch, 'test',
                        metrics['test_never_known_nodes '][args.loss], total_y,
                        'test_never_known_nodes')
        if args.scheduler == 'plateau':
            scheduler.step(metrics[metric_name][args.loss])
        elif args.scheduler == 'cosine' or args.scheduler == 'step':
            scheduler.step()

        print(f'-> Learning rate: {optimizer.param_groups[0]["lr"]:.5f}')
    return model, metrics, total_y


def better_metric(metric_a, metric_b):
    return metric_a < metric_b


def save_checkpoint(model, optimizer, args, epoch, mode, now_metric, total_y,
                    metric_name):
    checkpoint_path = args.best_checkpoint_path + f'best_{mode}_checkpoint.pth'
    best_metric = getattr(args, f'best_{mode}_metric')
    try:  # try-except here because checkpoint fname could be too long
        if (better_metric(now_metric, best_metric) or epoch == 0):
            setattr(args, f'best_{mode}_metric', now_metric)
            setattr(args, f'best_{mode}_metric_epoch', epoch)
            torch.save(
                {
                    'epoch': epoch,
                    'val_metric': now_metric,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            args.best_total_y.update({f'{mode}': total_y['test']})
            print(
                f'-> New best {metric_name} loss at epoch {epoch}! ({metric_name} loss: {now_metric:.4f})'
            )
    except Exception as e:
        print(e)


def train_model(model,
                optimizer,
                scheduler,
                data_loaders,
                criterions,
                args,
                wandb=None):

    if args.read_only:
        check_point_path = args.best_checkpoint_path + args.exp_id + '/'
        prediction_path = args.best_prediction_path + args.exp_id + '/'
        with open(prediction_path + 'best_total_y.pkl', 'rb') as f:
            args.best_total_y = pickle.load(f)
        read_out(data_loaders, check_point_path, criterions,
                 args.exp_id, args)
        check_point = check_point_path + 'best_train_checkpoint.pth'
        best_model_dict = torch.load(check_point)
        best_epoch = best_model_dict['epoch']
        print(f'Returning best train model from epoch {best_epoch}')
        model.load_state_dict(best_model_dict['state_dict'])
        return model

    # set file path
    args.best_checkpoint_path = args.best_checkpoint_path + args.exp_id + '/'
    if not os.path.exists(args.best_checkpoint_path):
        os.makedirs(args.best_checkpoint_path)
    args.best_prediction_path = args.best_prediction_path + args.exp_id + '/'
    if not os.path.exists(args.best_prediction_path):
        os.makedirs(args.best_prediction_path)
    args.fig_path = args.best_fig_path + args.exp_id + '/'
    if not os.path.exists(args.fig_path):
        os.makedirs(args.fig_path)

    args.best_val_metric = 1e10
    args.best_val_metric_epoch = -1
    args.best_train_metric = 1e10  # Interpolation / fitting also good to test
    args.best_train_metric_epoch = -1
    args.best_test_metric = 1e10
    args.best_test_metric_epoch = -1
    args.best_total_y = {}  # save best epoch test results
    print('-> Start training...')
    wait_count = 0
    for epoch in range(1, args.max_epoch + 1):
        print(f'-> Epoch {epoch} started...')
        st = time.time()
        _, metrics, _ = run_epoch(model, data_loaders, optimizer, scheduler,
                                  criterions, args, epoch)

        # Reset early stopping count if epoch improved
        if 'val' in args.stop_based:
            if args.best_val_metric_epoch == epoch or epoch < 20:
                wait_count = 0
            else:
                wait_count += 1
                print(f'-> Early stopping count: {wait_count}/{args.patience}')
        else:
            if args.best_train_metric_epoch == epoch or epoch < 20:
                wait_count = 0
            else:
                wait_count += 1
                print(f'-> Early stopping count: {wait_count}/{args.patience}')

        if wandb is not None:
            log_metrics = {}
            for mode in metrics.keys():
                for metric_name, metric_value in metrics[mode].items():
                    log_metrics[f'{mode}/{metric_name}'] = metric_value
            wandb.log(log_metrics, step=epoch)
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=epoch)

        print(f'├── Epoch {epoch} finished, took {time.time() - st:.2f}s')
        for mode in metrics:
            if 'train' in mode and 'val' in args.stop_based:
                continue
            if 'val' in mode and 'train' in args.stop_based:
                continue
            description = f'├── {mode} metrics:'
            for metric_name, metric in metrics[mode].items():
                description += f'{metric_name}: {metric:.4f}| '
            print(description)

        if wait_count == args.patience:
            print(f'Early stopping at epoch {epoch}!')
            break  # Exit for loop and do early stopping
    print('-> Training finished!')
    if 'train' in args.stop_based:
        print(
            f'-> Saved best train model checkpoint at epoch {args.best_train_metric_epoch}!'
        )
    elif 'val' in args.stop_based:
        print(
            f'-> Saved best val model checkpoint at epoch {args.best_val_metric_epoch}!'
        )
    print(
        f'-> Saved best test model checkpoint at epoch {args.best_test_metric_epoch}!'
    )
    print(f'   - Saved to: {args.best_checkpoint_path}')

    print("-> save predictions and metrics...")
    with open(args.best_prediction_path + 'best_total_y.pkl', 'wb') as f:
        args.best_total_y['always_known_nodes'], args.best_total_y[
            'never_known_nodes'], args.best_total_y[
                'new_add_nodes'], args.best_total_y[
                    'failed_nodes'] = data_loaders[
                        'test'].dataset.get_know_unknow_nodes()
        pickle.dump(args.best_total_y, f)
    read_out(data_loaders, args.best_checkpoint_path,
             criterions, args.exp_id, args)
    if 'val' in args.stop_based:
        check_point = args.best_checkpoint_path + 'best_val_checkpoint.pth'
    else:
        check_point = args.best_checkpoint_path + 'best_train_checkpoint.pth'
    best_model_dict = torch.load(check_point)
    epoch = best_model_dict['epoch']
    best_epoch = best_model_dict['epoch']
    print(f'Returning best train model from epoch {best_epoch}')
    model.load_state_dict(best_model_dict['state_dict'])
    return model
