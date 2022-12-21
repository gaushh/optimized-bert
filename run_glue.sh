pip install -r requirements.txt
export MODEL_TYPE="bert"
# model path should be same location as this shell file
export MODEL_PATH="bert-base-uncased"
export TASK_NAME="RTE"
export GLUE_DIR="data/glue_data"
export OUTPUT_DIR="output_glue"
export MAX_SEQ_LENGTH=128
export PER_GPU_TRAIN_BATCH_SIZE=32
export LEARNING_RATE=2e-5
export NUM_TRAIN_EPOCHS=3.0
#python src/glue/download_glue_data.py
python src/glue/run_glue.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_PATH \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME/ \
  --max_seq_length $MAX_SEQ_LENGTH \
  --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --output_dir $OUTPUT_DIR
