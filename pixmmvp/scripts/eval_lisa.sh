DATA=$1
OUT=$2
PROMPT=$3
ANS=$4

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate lisa
export PYTHONPATH="$PWD/../../LISA/":$PYTHONPATH
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

python ../inference/infer_lisa.py --root $DATA --root_images "$DATA/MMVP Images/" --version "xinlai/LISA-7B-v1-explanatory" --preds_dir "$OUT/preds_lisa_$PROMPTTEXT" --viz_dir "$OUT/viz_lisa_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_lisa_$PROMPTTEXT.jsonl" --precision='fp16' --load_in_8bit
