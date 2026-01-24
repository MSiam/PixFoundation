DATA=$1
OUT=$2
PROMPT=$3
ANS=$4
META=$5
POINT=$6
SAM_PATH=$7

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate llava
export PYTHONPATH="$PWD/../../LLaVA/":"$PWD/../../groundLMM/aas/":$PYTHONPATH
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

python ../inference/infer_pixfoundation.py --root $DATA --model-path "liuhaotian/llava-v1.5-7b" --preds_dir "$OUT/preds_llava-1.5-7b_$PROMPTTEXT" --viz_dir "$OUT/viz_llava-1.5-7b_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_llava-1.5-7b_$PROMPTTEXT.jsonl" --meta_file "$META/meta_llava-1.5-7b_spaCy_$PROMPTTEXT.txt" --sam-ckpt "${SAM_PATH}" --point_file "$POINT/point_llava-1.5-7b_$PROMPTTEXT.txt"

python ../inference/infer_pixfoundation.py --root $DATA --model-path "liuhaotian/llava-v1.5-13b" --preds_dir "$OUT/preds_llava-1.5-13b_$PROMPTTEXT" --viz_dir "$OUT/viz_llava-1.5-13b_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_llava-1.5-13b_$PROMPTTEXT.jsonl" --meta_file "$META/meta_llava-1.5-13b_spaCy_$PROMPTTEXT.txt" --sam-ckpt "${SAM_PATH}" --point_file "$POINT/point_llava-1.5-13b_$PROMPTTEXT.txt"
