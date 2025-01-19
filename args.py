import argparse


def initialize_args():
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument('--project_name', type=str, default='MoGERNN')

    # Model
    parser.add_argument('--model', type=str, default='MoGERNN')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--k_hop', type=int, default=1, help='k_hop for graph')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--e_layers',
                        type=int,
                        default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers',
                        type=int,
                        default=1,
                        help='num of decoder layers')
    parser.add_argument('--mean_expert',
                        action='store_true',
                        help='use mean expert')
    parser.add_argument('--weight_expert',
                        action='store_true',
                        help='use distane-based weighted expert')
    parser.add_argument('--max_expert',
                        action='store_true',
                        help='use max pooling expert')
    parser.add_argument('--min_expert',
                        action='store_true',
                        help='use min pooling expert')
    parser.add_argument('--diffusion_expert',
                        action='store_true',
                        help='use diffusion expert')
    parser.add_argument('--num_used_experts', type=int, default=3)

    parser.add_argument('--t_mark',
                        action='store_true',
                        help='do not add time stamp to feature')
    parser.add_argument('--pos_mark',
                        action='store_true',
                        help='do not add pos stamp to feature')

    # train
    parser.add_argument('--step_type',
                        type=str,
                        default='step',
                        help='step type for the model')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--loss',
                        type=str,
                        default='mse',
                        choices=[
                            'mse', 'mae', 'rmse', 'huber', 'msepinball',
                            'multipinball', 'nll'
                        ],
                        help='loss function used for backpropagation')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to train on')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='seed for reproducibility')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0001,
                        help='weight decay')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adamw',
                        choices=['adam', 'sgd', 'adamw'],
                        help='optimizer')
    parser.add_argument('--scheduler',
                        type=str,
                        default='plateau',
                        choices=['plateau', 'step', 'cosine', 'None'],
                        help='scheduler')
    parser.add_argument("--scheduler_patience",
                        type=int,
                        default=1,
                        help="scheduler patience")
    parser.add_argument('--patience',
                        type=int,
                        default=3,
                        help='early stopping epochs')
    parser.add_argument(
        '--stop_based',
        type=str,
        default='train_mask',
        choices=['val_total', 'val_mask', 'train_total', 'train_mask'],
        help='early stopping criteria')
    parser.add_argument('--return_best',
                        action='store_true',
                        help='return best model')
    parser.add_argument('--test_for_changed',
                        action='store_true',
                        help='test pretrained model in changed sensor network')
    parser.add_argument('--read_only', action='store_true', help='read only')
    parser.add_argument('--inference_only',
                        action='store_true',
                        help='inference only')
    parser.add_argument('--pretrained_model_path',
                        default=None,
                        help='path to pretrained model')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_scale', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)

    # task
    parser.add_argument('--input_len', type=int, required=True)
    parser.add_argument('--pred_len', type=int, required=True)
    parser.add_argument('--look_back', type=int, default=0)
    parser.add_argument('--slide_step',
                        type=int,
                        default=1,
                        help='step for sliding window')
    parser.add_argument('--target_features',
                        type=str,
                        required=True,
                        choices=['q', 'o', 'v', 'qv', 'qov'],
                        help='q is flow, v is speed, o is occupancy')
    parser.add_argument('--input_features',
                        type=str,
                        required=True,
                        choices=['q', 'o', 'v', 'qv', 'qov'],
                        help='q is flow, v is speed, o is occupancy')
    parser.add_argument('--train_ratio',
                        type=float,
                        default=0.6,
                        help='train ratio')
    parser.add_argument('--valid_ratio',
                        type=float,
                        default=0.2,
                        help='validation ratio')

    parser.add_argument(
        '--inverse',
        action='store_true',
        help='inverse the data for better results presentation')

    # save and logging settings
    parser.add_argument('--wandb',
                        action='store_true',
                        help='enable wandb logging')
    parser.add_argument('--wandb_entity',
                        type=str,
                        default='zqslalala',
                        help='user name of wandb')
    parser.add_argument('--best_checkpoint_path',
                        type=str,
                        default='./check_points/',
                        help='best val checkpoint path')
    parser.add_argument('--best_prediction_path',
                        type=str,
                        default='./predictions/',
                        help='results path')
    parser.add_argument('--best_metrics_path',
                        type=str,
                        default='./',
                        help='results path')
    parser.add_argument('--best_fig_path',
                        type=str,
                        default='./figures/',
                        help='figures path')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./logs/',
                        help='log path')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--log_interval', type=int, default=100)

    parser.add_argument('--save_model',
                        action='store_true',
                        help='save model not model state dict')
    # dataset
    parser.add_argument(
        '--unknown_nodes_path',
        type=str,
        help='path for unobserved sensors id in the train data')
    parser.add_argument('--unknown_nodes_list',
                        type=str,
                        default="",
                        help='unobserved sensors in the train data')
    parser.add_argument(
        '--unknown_nodes_num',
        type=int,
        default=0,
        help='number of unobserved sensors  in the training data')
    parser.add_argument('--ori_unknown_nodes_list',
                        type=str,
                        default="",
                        help='unobserved sensors in the original data')
    parser.add_argument(
        '--dloader_name',
        type=str,
        default='METRLA',
        choices=['METRLA', 'PEMSBAY', 'METRLA-dynamic', 'PEMSBAY-dynamic'],
        help='dynamic denotes the scenario where the scensor is changing')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--freq', type=str, default='min')

    args = parser.parse_args()
    args.unknown_nodes_list = [
        int(x) for x in args.unknown_nodes_list.split(',') if x
    ]
    args.ori_unknown_nodes_list = [
        int(x) for x in args.ori_unknown_nodes_list.split(',') if x
    ]
    args.unknown_nodes_list = [
        x for x in args.unknown_nodes_list
        if x not in args.ori_unknown_nodes_list
    ]
    assert args.input_features in args.target_features, 'input_features should be in target_features'
    args.c_out = len(args.target_features)
    args.enc_in = len(args.input_features)
    args.enc_mark_in = 3 * (args.t_mark) + 2 * (args.pos_mark)
    print(
        f'enc_in enc_mark_in, and c_out is set automatically based on input_features,target_features and t_mark,pos_mark',
    )
    args.num_experts = 1 * (args.mean_expert) + 1 * (
        args.weight_expert) + 1 * (args.max_expert) + 1 * (
            args.min_expert) + 1 * (args.diffusion_expert)
    return args
