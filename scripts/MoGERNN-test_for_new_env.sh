infea=v
tarfea=v
mepoch=200
bs=64
lr=0.001
patience=10
d_model=64
model=MoGERNN

data_path='./data/METR-LA/raw_data.pkl'


# apply pretrained model in changed sensor network
python main.py  --model $model --max_epoch $mepoch --lr $lr \
                --input_features $infea --target_features $tarfea  --d_model $d_model  \
                --input_len 12 --pred_len 12 --look_back 0 --dloader_name METRLA --batch_size $bs \
                --loss mse --data_path $data_path --num_workers 8 \
                --patience $patience --k_hop 1 --unknown_nodes_path './data/METR-LA/unknown_nodes.npy' \
                --slide_step 12 --scheduler None \
                --mean_expert --weight_expert --max_expert --min_expert --diffusion_expert \
                --stop_based 'val_mask'  --seed 32 --num_used_experts 3 --test_for_changed \
                --test_for_changed
