DATA=$1
OUT=$2
PROMPT=$3
ANS=$4
META=$5

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate llava
export PYTHONPATH="LLaVA/":"$PWD/../../groundLMM/aas/":$PYTHONPATH # Modify Llava path to your cloned repository path
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

python ../inference/infer_pixfoundation.py --root $DATA --model-path "liuhaotian/llava-v1.5-7b" --preds_dir "$OUT/preds_llava-1.5-7b-liu_$PROMPTTEXT" --viz_dir "$OUT/viz_llava-1.5-7b-liu_$PROMPTTEXT" --prompt_for_seg $PROMPT --answers_file "$ANS/answers_llava-1.5-7b-liu_$PROMPTTEXT.jsonl" --meta_file "$META/meta_llava-1.5-7b-liu_spaCy_$PROMPTTEXT.txt" --sam-ckpt "sam_checkpoints/sam_vit_h_4b8939.pth"

