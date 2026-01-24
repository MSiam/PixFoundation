DATA=$1
OUT=$2
PROMPT=$3
ANS=$4

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate llavag
export PYTHONPATH="$PWD/../../LLava-Grounding/":$PYTHONPATH
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

python ../inference/infer_llavag.py --directory $DATA --path_vision_cfg "../../LLava-Grounding/configs/openseed/openseed_swint_lang_joint_2st_visual_prompt.yaml" --path_inter_cfg "../../LLava-Grounding/configs/semsam/visual_prompt_encoder.yaml" --model_path "Haozhangcx/llava_grounding_gd_vp" --preds_dir "$OUT/preds_llavag_$PROMPTTEXT" --viz_dir "$OUT/viz_llavag_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_llavag_$PROMPTTEXT.jsonl"
