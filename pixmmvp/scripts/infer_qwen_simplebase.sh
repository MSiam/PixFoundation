DATA=$1
OUT=$2
PROMPT=$3
PROMPTTYPE=$4
ANS=$5
SAM_PATH=$6

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate qwen
export PYTHONPATH="$PWD/../../Qwen3-VL/":$PYTHONPATH
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

python ../inference/infer_qwen_simplebaseline.py --model_path "Qwen/Qwen2.5-VL-7B-Instruct"  --directory $DATA --preds_dir "$OUT/preds_qwen_simplebaseline_${PROMPTTYPE}_${PROMPTTEXT}" --viz_dir "$OUT/viz_qwen_simplebaseline_${PROMPTTYPE}_${PROMPTTEXT}" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_qwen_simplebaseline_${PROMPTTYPE}_${PROMPTTEXT}.jsonl" --sam_ckpt "${SAM_PATH}" --prompt_type ${PROMPTTYPE}
