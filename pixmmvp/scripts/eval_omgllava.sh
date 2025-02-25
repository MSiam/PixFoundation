DATA=$1
OUT=$2
PROMPT=$3
ANS=$4

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate omgllava
export PYTHONPATH="$PWD/../../OMG-Seg/omg_llava/":$PYTHONPATH
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

python ../inference/infer_omgllava.py ../../OMG-Seg/omg_llava/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py pretrained_omgllava/omg_llava/omg_llava_7b_finetune_8gpus.pth --root $DATA --root_images "$DATA/MMVP Images/" --preds_dir "$OUT/preds_omgllava_$PROMPTTEXT" --viz_dir "$OUT/viz_omgllava_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_omgllava_$PROMPTTEXT.jsonl" --model_path pretrained_omgllava/
