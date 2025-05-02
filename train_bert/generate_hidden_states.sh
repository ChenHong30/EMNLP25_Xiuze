echo $1
data_path=./data/
data_name=/reviews.pickle
# d_type=ClothingShoesAndJewelry
d_type=MoviesAndTV
d_index=1
llm_model=/hpc2hdd/home/hchen763/jhaidata/local_model/gpt2
split=valid

CUDA_VISIBLE_DEVICES=$1 python -u ./train_bert/generate_hidden_states.py \
    --data_path ${data_path}${d_type}${data_name} \
    --index_dir ${data_path}${d_type}\/${d_index} \
    --llm_model ${llm_model} \
    --output_dir ./train_bert/bert_finetune_data/${d_type}/${split} \
    --split ${split} \
    --batch_size 2048 \
    --words 20 \
    --cuda