# Initialize default values
DATA_NAME="zebra-grid"
# FU_MODE="self_verification"
# FU_MODE="neg_feedback"
# FU_MODE="zebra_oracle"
FU_MODE="neg_feedback_v2"

T=2
# model_name="openai/gpt-4o-mini-2024-07-18"
# model_pretty_name="gpt-4o-mini-2024-07-18.${FU_MODE}.T=${T}"
model_name="openai/gpt-4o-2024-08-06"
model_pretty_name="gpt-4o-2024-08-06.${FU_MODE}.T=${T}"




n_shards=8
run_name="${FU_MODE}"
TEMP=0
TOP_P=1.0
rp=1.0
engine_name="openai"
MAX_TOKENS=4096; 
num_outputs=1  # New default value
batch_size=4; 
CACHE_DIR=${HF_HOME:-"default"}


# Check if required arguments are provided
if [ -z "$DATA_NAME" ] || [ -z "$model_name" ] || [ -z "$model_pretty_name" ] || [ -z "$n_shards" ]; then
  echo "Usage: $0 -d DATA_NAME -m model_name -p model_pretty_name -s n_shards [-r run_name] [-t TEMP] [-o TOP_P] [-e rp] [-f engine_name] [-n num_outputs]"
  exit 1
fi

# output_dir="result_dirs/${DATA_NAME}/cot=${cot}/" 
if [ "$run_name" = "default" ]; then
    output_dir="result_dirs/${DATA_NAME}/" 
else
    output_dir="result_dirs/${DATA_NAME}/${run_name}/" 
fi


echo "Using Data-parallelism"
shards_dir="${output_dir}/tmp_${model_pretty_name}"
for ((shard_id = 0; shard_id < $n_shards; shard_id++)); do
    python src/unified_infer.py \
        --follow_up_mode "${FU_MODE}" \
        --follow_up_file "result_dirs_follow_up/zebra-grid/${model_pretty_name}.json" \
        --num_shards $n_shards \
        --shard_id $shard_id \
        --data_name $DATA_NAME \
        --engine $engine_name \
        --model_name $model_name \
        --run_name $run_name \
        --model_pretty_name $model_pretty_name \
        --top_p $TOP_P --temperature $TEMP --repetition_penalty $rp \
        --batch_size $batch_size --max_tokens $MAX_TOKENS \
        --num_outputs $num_outputs \
        --output_folder $shards_dir/ \
        &
done 
wait 
python src/merge_results.py $shards_dir/ $model_pretty_name
cp $shards_dir/${model_pretty_name}.json $output_dir/${model_pretty_name}.json
 