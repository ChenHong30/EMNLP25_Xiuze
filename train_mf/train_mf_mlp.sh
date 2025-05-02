echo $1
seed=1111
model_name=/MF/
data_path=./data/
data_name=/reviews.pickle
checkpoint_dir=./models/
out_log_dir=./logs/
out_log=train.log

for d_type in TripAdvisor ClothingShoesAndJewelry MoviesAndTV Yelp  # data
do
    for d_index in 1 2 3 4 5  # data_index
    do
        mkdir -p ${out_log_dir}${d_type}\/${d_index}${model_name}
        mkdir -p ${checkpoint_dir}${d_type}\/${d_index}${model_name}

        echo "data_type: $d_type, data_index: $d_index , seed: $seed"
        TRANSFORMERS_CACHE=./llm/ \
        HF_DATASETS_CACHE=./llm/ \
        CUDA_VISIBLE_DEVICES=$1 D:/Anaconda3/envs/torch2.2/python -u ./train_mf_mlp.py \
            -data_path ${data_path}${d_type}${data_name} \
            -index_dir ${data_path}${d_type}\/${d_index}\/ \
            -model_name mf \
            -emsize 64 \
            -lr 0.001 \
            -epochs 100 \
            -batch_size 2048 \
            -seed $seed \
            -cuda \
            -log_interval 200 \
            -checkpoint ${checkpoint_dir}${d_type}\/${d_index}${model_name} \
            -endure_times 5 \
            > ${out_log_dir}${d_type}\/${d_index}${model_name}${out_log}
    done
done
