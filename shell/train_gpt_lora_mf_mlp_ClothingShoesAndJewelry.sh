echo $1
seed=1111
model_name=/GPTLoRA/
data_path=./data/
data_name=/reviews.pickle
llm_model=/hpc2hdd/home/hchen763/jhaidata/local_model/gpt2
pre_model=/MF/model.pt
checkpoint_dir=./models/
out_file_dir=./outputs/
out_file_name=generated-gpt2-clip.txt
out_log_dir=./logs/
out_log=train-clip.log

for d_type in ClothingShoesAndJewelry
do
    for d_index in 1 2 3 4 5  # data_index
    do
        mkdir -p ${out_log_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${out_file_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${checkpoint_dir}${d_type}\/${d_index}${model_name}

        echo "data_type: $d_type, data_index: $d_index , seed: $seed"
        TRANSFORMERS_CACHE=./llm/ \
        HF_DATASETS_CACHE=./llm/ \
        CUDA_VISIBLE_DEVICES=$1 python -u ./train_gpt_lora_mf_mlp.py \
            -data_path ${data_path}${d_type}${data_name} \
            -index_dir ${data_path}${d_type}\/${d_index}\/ \
            -llm_model ${llm_model} \
            -pre_model ${checkpoint_dir}${d_type}\/${d_index}${pre_model} \
            -post_att \
            -lora_nums 0,1,2,3,4,5,6,7,8,9,10,11 \
            -lora_dim 8 \
            -nhead 8 \
            -lr 0.0001 \
            -epochs 100 \
            -batch_size 128 \
            -seed $seed \
            -cuda \
            -log_interval 200 \
            -checkpoint ${checkpoint_dir}${d_type}\/${d_index}${model_name} \
            -outf ${out_file_dir}${d_type}\/${d_index}${model_name}${out_file_name} \
            -endure_times 5 \
            -rating_reg 0.1 \
            -text_reg 1 \
            -words 20 \
            > ${out_log_dir}${d_type}\/${d_index}${model_name}${out_log}
    done
done
