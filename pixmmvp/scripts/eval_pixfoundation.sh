MODELTEXT=$1
DATA=$2
PRED=$3
VIZ=$4
PROMPT=$5
META=$6
TYPE=$7

if [ $PROMPT == "1" ]
then
    PROMPTTEXT="protocol3"
elif [ $PROMPT == "0" ]
then
    PROMPTTEXT="protocol1"
else
    PROMPTTEXT="protocol2"
fi

mkdir $VIZ
echo "${META}meta_${MODELTEXT}_spaCy_${PROMPTTEXT}.txt"

if [ $TYPE == "oracle" ]
then
    python ../eval/eval_nonpixLMM_iou.py --preds_dir "${PRED}preds_${MODELTEXT}_${PROMPTTEXT}"  --dataset_root "${DATA}" --viz_dir "${VIZ}/viz_${MODELTEXT}_${PROMPTTEXT}_${TYPE}/" --type $TYPE
elif [ $TYPE == "spacy_score" ]
then
    python ../eval/eval_nonpixLMM_iou.py --preds_dir "${PRED}preds_${MODELTEXT}_${PROMPTTEXT}"  --dataset_root "${DATA}" --viz_dir "${VIZ}/viz_${MODELTEXT}_${PROMPTTEXT}_${TYPE}/" --meta_file "${META}meta_${MODELTEXT}_spaCy_${PROMPTTEXT}.txt" --type $TYPE
else
    python ../eval/eval_nonpixLMM_iou.py --preds_dir "${PRED}preds_${MODELTEXT}_${PROMPTTEXT}"  --dataset_root "${DATA}" --viz_dir "${VIZ}/viz_${MODELTEXT}_${PROMPTTEXT}_${TYPE}/" --meta_file "${META}meta_${MODELTEXT}_gptauto_${PROMPTTEXT}.jsonl" --type $TYPE
fi
