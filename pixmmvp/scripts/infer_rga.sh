DATA=$1
OUT=$2
PROMPT=$3
ANS=$4
MODE=$5
POSTPROC=$6

NUM_FRMS_MLLM=1
MAX_PIXELS=$((384*28*28))

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate rga
export PYTHONPATH="$PWD/../../RGA3-release/":$PYTHONPATH
echo $PYTHONPATH

# Protocol1 : Inquiry about the question with choices directly 
# Protocol2: same but withinstruction to generate one option letter
# Protocol3: identify referring expression of the object of itnerest from the question

if [ $PROMPT == "1" ]
then
    PROMPTTEXT="protocol3"
elif [ $PROMPT == "0" ]
then
    PROMPTTEXT="protocol1"
else
    PROMPTTEXT="protocol2"
fi

CUDA_VISIBLE_DEVICES=0 python ../inference/infer_rga.py \
  --dataset_root "$DATA" \
  --version "SurplusDeficit/UniGR-7B" \
  --preds_dir "$OUT/preds_rga_${PROMPTTEXT}_${MODE}_${POSTPROC}" \
  --viz_dir "$OUT/viz_rga_${PROMPTTEXT}_${MODE}_${POSTPROC}" \
  --num_frames_mllm $NUM_FRMS_MLLM \
  --max_pixels $MAX_PIXELS \
  --prompt_for_seg $PROMPT \
  --answers_file "$ANS/answers_rga_${PROMPTTEXT}_${MODE}_${POSTPROC}.jsonl" \
  --inference_mode ${MODE} \
  --postproc ${POSTPROC}
