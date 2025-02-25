DATA=$1
OUT=$2
PROMPT=$3
ANS=$4

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate glamm
export PYTHONPATH="$PWD/../../groundingLMM/":$PYTHONPATH
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

python ../inference/infer_glamm.py --root $DATA --root_images "$DATA/MMVP Images/" --hf_model_path "MBZUAI/GLaMM-FullScope" --preds_dir "$OUT/preds_glammfull_$PROMPTTEXT" --viz_dir "$OUT/viz_glammfull_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_glammfull_$PROMPTTEXT.jsonl"

python ../inference/infer_glamm.py --root $DATA --root_images "$DATA/MMVP Images/" --hf_model_path "MBZUAI/GLaMM-RegCap-RefCOCOg" --preds_dir "$OUT/preds_glammregcap_$PROMPTTEXT" --viz_dir "$OUT/viz_glammregcap_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_glammregcap_$PROMPTTEXT.jsonl"
