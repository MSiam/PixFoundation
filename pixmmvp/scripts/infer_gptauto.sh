MODELTEXT=$1
DATA=$2
PRED=$3
PROMPT=$4
META=$5
OPENAIKEY=$6

if [ $PROMPT == "1" ]
then
    PROMPTTEXT="protocol3"
elif [ $PROMPT == "0" ]
then
    PROMPTTEXT="protocol1"
else
    PROMPTTEXT="protocol2"
fi

echo $PROMPTTEXT

# Output the highlighted images with masks
python ../inference/infer_pixfoundation_auto_gpt.py --root "$DATA" --preds_dir "${PRED}/preds_${MODELTEXT}_${PROMPTTEXT}/" --auto_vis_dir "${PRED}/autoviz_${MODELTEXT}_${PROMPTTEXT}/" --stage 1

# Prompt GPT to judge the best segmentation (Mask Selection)
python ../inference/infer_pixfoundation_auto_gpt.py --root "$DATA" --auto_vis_dir "${PRED}/autoviz_${MODELTEXT}_${PROMPTTEXT}/" --stage 2 --openai_api_key ${OPENAIKEY} --answers_file "${META}/meta_${MODELTEXT}_gptauto_${PROMPTTEXT}.jsonl" --preds_dir "${PRED}/preds_${MODELTEXT}_${PROMPTTEXT}/"
