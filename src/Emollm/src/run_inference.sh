#export CUDA_VISIBLE_DEVICES='2'
export ABS_PATH="local_path"
#CPU Optimization Warnings (oneDNN)
export TF_ENABLE_ONEDNN_OPTS=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



model_name_or_path=lzw1008/Emollama-chat-7b # checkpoint

#Emollm input
#Liar_data set input
#infer_file=data/liar_test_senti.json
#predict_file=predict_liar_test_senti.json
#pheme_data set input
infer_file=data/pheme_veracity_labels_senti.json
predict_file=predict_pheme_veracity_labels_senti.json

# inference
python src/inference.py \
    --model_name_or_path $model_name_or_path \
    --infer_file $infer_file \
    --predict_file $predict_file \
    --batch_size 32 \
    --seed 123
    #--llama \

