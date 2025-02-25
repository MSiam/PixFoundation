DATA=$1
OUT=$2
PROMPT=$3
ANS=$4
META=$5

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

python ../inference/infer_pixfoundation_cambrian.py --root $DATA --model-path "nyu-visionx/cambrian-8b" --preds_dir "$OUT/preds_cambrian-8b_$PROMPTTEXT" --viz_dir "$OUT/viz_cambrian-8b_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_cambrian-8b_$PROMPTTEXT.jsonl" --meta_file "$META/meta_cambrian-8b_spaCy_$PROMPTTEXT.txt" --sam-ckpt "sam_checkpoints/sam_vit_h_4b8939.pth"
