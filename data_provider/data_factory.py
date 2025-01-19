from data_provider.graph_data_loader import StaticPEMS, DynamicPEMS
from torch.utils.data import DataLoader


def data_provider(args, flag):
    if args.test_for_changed:
        Dataset = DynamicPEMS
    else:
        Dataset = StaticPEMS
    if flag != 'train':
        shuffle_flag = False
        drop_last = False
        batch_size = 256
        # if not (args.test_for_changed
        #  or args.inference_only) else 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    data_set = Dataset(data_path=args.data_path,
                       size=[args.input_len, args.look_back, args.pred_len],
                       flag=flag,
                       scale=not args.no_scale,
                       freq=freq,
                       inverse=args.inverse,
                       target_features=args.target_features,
                       input_features=args.input_features,
                       dloader_name=args.dloader_name,
                       step=args.slide_step,
                       splits=[args.train_ratio, args.valid_ratio],
                       unknown_nodes_path=args.unknown_nodes_path,
                       unknown_nodes_num=args.unknown_nodes_num,
                       unknown_nodes_list=args.unknown_nodes_list)

    return DataLoader(data_set,
                      batch_size=batch_size,
                      shuffle=shuffle_flag,
                      num_workers=args.num_workers,
                      drop_last=drop_last)
