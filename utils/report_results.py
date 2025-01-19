import pickle
import numpy as np
import csv
import os


def load_prediction(path, mode):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        y = data[mode]['true_trans']
        y_pred = data[mode]['pred_trans']
        AAS = data['always_known_nodes']
        VS = data['never_known_nodes']
        FS = data['failed_nodes']
        NAS = data['new_add_nodes']
        return y_pred, y, AAS, VS, FS, NAS


def report_error(y_pred, y, AAS, VS, FS, NAS, model_name, reports_file: str):
    # assume y_pred and y shape are batc_size,time_seq_len,num_nodes,feature
    # caculte the loss on each time step
    nodes = {'AAS': AAS, 'VS': VS, 'FS': FS, 'NAS': NAS}
    y_pred = y_pred.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    y_omi = np.isnan(y)
    csv_head = ['model_name', 'node_type']
    error_heads = [[
        f'mape_at_{x*5}min', f'mae_at_{x*5}min', f'rmse_at_{x*5}min'
    ] for x in [3, 6, 12]]
    for error_head in error_heads:
        csv_head += error_head
    error_heads = [[
        f'mape_at_{x*5}min', f'mae_at_{x*5}min', f'rmse_at_{x*5}min'
    ] for x in range(1, y.shape[1] + 1)]
    for error_head in error_heads:
        csv_head += error_head
    # 确保文件路径存在
    if not os.path.isfile(reports_file):
        os.makedirs(os.path.dirname(reports_file), exist_ok=True)
        with open(reports_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(csv_head)
    csv_contents = []
    for node_type, node in nodes.items():
        if len(node) == 0:
            continue
        y_part = y[:, :, node, :]
        y_pred_part = y_pred[:, :, node, :]
        y_omi_part = y_omi[:, :, node, :]
        print(
            'all_mape:',
            np.mean(
                abs(y_part[~y_omi_part] - y_pred_part[~y_omi_part]) /
                (y_part[~y_omi_part])))
        # mape
        ape = np.abs(y_part - y_pred_part) / (y_part)
        mape = np.round(np.nanmean(ape, axis=(0, 2, 3)) * 100, 2)
        # mae
        ae = np.abs(y_part - y_pred_part)
        mae = np.round(np.nanmean(ae, axis=(0, 2, 3)), 2)
        # rmse
        se = np.square(y_part - y_pred_part)
        mse = np.nanmean(se, axis=(0, 2, 3))
        rmse = np.round(np.sqrt(mse), 2)
        csv_contents += [model_name, node_type]
        for i in [2, 5, 11]:
            csv_contents += [mape[i], mae[i], rmse[i]]
        for i in range(len(mape)):
            csv_contents += [mape[i], mae[i], rmse[i]]
        with open(reports_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(csv_contents)
        csv_contents = []


if __name__ == '__main__':
    path = 'predictions/MoGERNN-METRLA-12-12-0-12-v-v-t_mark-False-pos_mark-False-seed-32/best_total_y.pkl'
    y_pred, y, AAS, VS, FS, NAS, model_name, *args = load_prediction(
        path, 'val')
    report_error(y_pred, y, AAS, VS, FS, NAS, model_name, './reports.csv')
    path = 'predictions/MoGERNN-PEMSBAY-12-12-0-12-v-v-t_mark-False-pos_mark-False-seed-42/best_total_y.pkl'
    y_pred, y, AAS, VS, FS, NAS, model_name, *args = load_prediction(
        path, 'val')
    report_error(y_pred, y, AAS, VS, FS, NAS, model_name, './reports.csv')
