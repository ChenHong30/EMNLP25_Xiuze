echo $1
seed=1111
model_name=/GPTLoRA/
data_path=./data/
data_name=/reviews.pickle
llm_model=./llm/models-gpt2
checkpoint_dir=./models/
out_file_dir=./outputs/
out_file_name=generated.txt
out_log_dir=./logs/
out_log=train.log

for d_type in TripAdvisor
do
    for d_index in 1
    do
        for n_lr in 0.0001 0.001 0.005 0.01 # lr
        do
            mkdir -p ${out_log_dir}${d_type}\/${d_index}\/${n_lr}${model_name}
            mkdir -p ${out_file_dir}${d_type}\/${d_index}\/${n_lr}${model_name}
            mkdir -p ${checkpoint_dir}${d_type}\/${d_index}\/${n_lr}${model_name}

            echo "data_type: $d_type, data_index: $d_index , seed: $seed, n_lr: $n_lr"
            TRANSFORMERS_CACHE=./llm/ \
            HF_DATASETS_CACHE=./llm/ \
            CUDA_VISIBLE_DEVICES=$1 D:/Anaconda3/envs/torch2.2_rec/python -u ./train_gpt_lora.py \
                -data_path ${data_path}${d_type}${data_name} \
                -index_dir ${data_path}${d_type}\/${d_index}\/ \
                -llm_model ${llm_model} \
                -lora_nums 0,1,2,3,4,5,6,7,8,9,10,11 \
                -lora_dim 8 \
                -nhead 8 \
                -lr $n_lr \
                -epochs 100 \
                -batch_size 128 \
                -seed $seed \
                -cuda \
                -log_interval 200 \
                -checkpoint ${checkpoint_dir}${d_type}\/${d_index}\/${n_lr}${model_name} \
                -outf ${out_file_dir}${d_type}\/${d_index}$\/${n_lr}${model_name}${out_file_name} \
                -endure_times 5 \
                -rating_reg 0.01 \
                -text_reg 1 \
                -words 20 \
                > ${out_log_dir}${d_type}\/${d_index}\/${n_lr}${model_name}${out_log}
        done
    done
done
