echo $1
d_type=MoviesAndTV
# bert_model_name=/hpc2hdd/home/hchen763/jhaidata/local_model/bert-base-uncased
bert_model_name=google/bert_uncased_L-2_H-128_A-2
lr=5e-5
epochs=100
patience=5

CUDA_VISIBLE_DEVICES=$1 python -u ./train_bert/finetune_bert_on_states.py \
    --train_data_dir ./train_bert/bert_finetune_data/${d_type}/train/ \
    --valid_data_dir ./train_bert/bert_finetune_data/${d_type}/valid/ \
    --bert_model_name ${bert_model_name} \
    --output_model_dir ./train_bert/fine_tuned_bert/${d_type}/ \
    --epochs ${epochs} \
    --lr ${lr} \
    --patience ${patience} \
    --cuda
python /hpc2hdd/home/hchen763/test.py