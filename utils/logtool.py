import rich.syntax
import rich.tree
import os
from omegaconf import OmegaConf, DictConfig, ListConfig
from datetime import datetime
import pytz


def get_formatted_time(timezone='Etc/GMT-1'):
    # 创建一个datetime对象
    dt = datetime.now()
    if timezone is None:
        return dt.strftime("-%m-%d-%H-%M-%S-")
    tz = pytz.timezone(timezone)
    dt = dt.astimezone(tz)
    return dt.strftime("-%m-%d-%H-%M-%S-")


def print_header(x, ):
    print('-' * len(x))
    print(x)
    print('-' * len(x))


def print_epoch_metrics(metrics):
    for mode in metrics.keys():
        print('-' * 4, f'{mode}', '-' * 4)
        for k, v in metrics[mode].items():
            print(f'- {k}: {v:.5f}')


def print_train(args):
    print_header('*** Training Configurations ***')
    print(
        f'├── Input/Look Back/Prediction/Slide Horizon: {args.input_len,args.look_back,args.pred_len,args.slide_step}'
    )
    print(
        f'├── Dims: input={args.enc_in}, mark={args.enc_mark_in}, model_dim={args.d_model}, output={args.c_out}'
    )
    print(f'├── Number of trainable parameters: {args.num_params }')
    print(f'├── Experiment name: {args.exp_id}')
    print(f'├── Checkpoints save to: {args.best_checkpoint_path}')
    print(f'├── Predictions save to: {args.best_prediction_path}')
    print(f'├── Metrics save to: {args.best_metrics_path}')


def set_wandb(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if args.wandb:
        project_name = f'{args.project_name}-{args.dloader_name}-{args.input_len}|{args.look_back}|{args.pred_len}|{args.slide_step}'
        import wandb
        wandb.require("core")
        run_name = args.exp_id
        wandb.init(
            config={},
            entity=args.wandb_entity,  # user id
            name=run_name,  #experiment name
            project=project_name,
            dir=args.log_dir)
        wandb.config.update(args)  # 记录实验参数
        return wandb
    return None


def print_args(args, return_dict=False, verbose=True):
    attributes = [a for a in dir(args) if a[0] != '_']
    arg_dict = {}  # switched to ewr
    if verbose: print('ARGPARSE ARGS')
    for ix, attr in enumerate(attributes):
        # fancy = '└──' if ix == len(attributes) - 1 else '├──'
        fancy = '└─' if ix == len(attributes) - 1 else '├─'
        if verbose: print(f'{fancy} {attr}: {getattr(args, attr)}')
        arg_dict[attr] = getattr(args, attr)
    if return_dict:
        return arg_dict


def print_config(config, resolve: bool = True, name: str = 'CONFIG') -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    tree = rich.tree.Tree(
        name,
        style="bold red",
    )

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style="bold black")

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)
        elif isinstance(config_section, ListConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"), style='red')

    rich.print(tree)


def detail_metrics(dataloader,
                   criterions,
                   args,
                   mode,
                   metrics,
                   y_omi,
                   loss,
                   inverse_y,
                   inverse_y_hat,
                   max_bacth_num=None,
                   print_enabled=False,
                   batch_ix=0):
    max_bacth_num = len(dataloader) if max_bacth_num is None else max_bacth_num
    description = f'└── {mode} batch {batch_ix+1}/{max_bacth_num}'
    for metric_name, criterion in criterions.items():
        if metric_name == args.loss:
            metric_value = loss.item()
        elif metric_name == 'split_phy_loss':
            metric_value = criterion(
                y_pred=inverse_y_hat,
                y_true=inverse_y,
                miss_indicator=y_omi,
            )
        else:
            raise NotImplementedError
        if metric_name == 'split_phy_loss':
            for sub_metric_name, sub_metric_value in metric_value.items():
                metrics[sub_metric_name] += sub_metric_value
        else:
            metrics[metric_name] += metric_value  #epoch_loss

        if ((batch_ix + 1) % args.log_interval == 0
                and print_enabled) or batch_ix == max_bacth_num - 1:
            if metric_name == 'split_phy_loss':
                if 'train' in mode:
                    for sub_metric_name, sub_metric_value in metric_value.items(
                    ):
                        description += f' | {sub_metric_name}: {(metrics[sub_metric_name]/(batch_ix+1)):.5f}'
                else:
                    for sub_metric_name, sub_metric_value in metric_value.items(
                    ):
                        description += f' | {sub_metric_name}: {sub_metric_value:.5f}'
            else:
                if 'train' in mode:
                    description += f' | {metric_name}: {(metrics[metric_name]/(batch_ix+1)):.5f}'
                else:
                    description += f' | {metric_name}: {metric_value:.5f}'

    if ((batch_ix + 1) % args.log_interval == 0
            and print_enabled) or (batch_ix == max_bacth_num - 1
                                   and print_enabled):
        print(description)
    return metrics
