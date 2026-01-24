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
conda activate cambrian
export PYTHONPATH="$PWD/../../cambrian/":"$PWD/../../groundLMM/aas/":$PYTHONPATH
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

python ../inference/infer_simple_baseline_cambrian.py --root $DATA --model-path "nyu-visionx/cambrian-8b" --preds_dir "$OUT/preds_cambrian-8b_simplebaseline_$PROMPTTEXT" --viz_dir "$OUT/viz_cambrian-8b_simplebaseline_$PROMPTTEXT" --answers_file "$ANS/answers_cambrian-8b_simplebaseline_$PROMPTTEXT.jsonl" --sam-ckpt "${SAM_PATH}"
