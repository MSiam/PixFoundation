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

if [ $PROMPT == "0" ] || [ $PROMPT == 3 ] 
then
    python ../inference/infer_omgllava.py ../../OMG-Seg/omg_llava/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py OMG_WEIGHTS_PATH/omg_llava_7b_finetune_8gpus.pth --root "$DATA/CV-Bench/" --root_images $DATA --preds_dir "$OUT/preds_omgllava_$PROMPTTEXT" --viz_dir "$OUT/viz_omgllava_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_omgllava_ade_$PROMPTTEXT.jsonl" --model_path OMG_WEIGHTS_PATH --cvbench_section "ADE20K"
    
    python ../inference/infer_omgllava.py ../../OMG-Seg/omg_llava/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py OMG_WEIGHTS_PATH/omg_llava_7b_finetune_8gpus.pth --root "$DATA/CV-Bench/" --root_images $DATA --preds_dir "$OUT/preds_omgllava_$PROMPTTEXT" --viz_dir "$OUT/viz_omgllava_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_omgllava_coco_$PROMPTTEXT.jsonl" --model_path OMG_WEIGHTS_PATH --cvbench_section "COCO"

elif [ $PROMPT == "1" ]
then

    python ../inference/infer_omgllava.py ../../OMG-Seg/omg_llava/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py OMG_WEIGHTS_PATH/omg_llava_7b_finetune_8gpus.pth --root "$DATA/CV-Bench/" --root_images $DATA --preds_dir "$OUT/preds_omgllava_$PROMPTTEXT" --viz_dir "$OUT/viz_omgllava_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_omgllava_$PROMPTTEXT.jsonl" --model_path OMG_WEIGHTS_PATH --cvbench_section "ALL"

fi
