###################################################### Pixel-level MLLMs ##########################################################
# OMG Llava Example
# Set the following variables, and modify OMGLLAVACONDA to your environment setup and API_KEY to OpenAI API Key.
# Similar setup to GLAMM (eval_glamm.sh), LISA (eval_lisa.sh), Llava-G (eval_llavag.sh).
DATA="data/pixmmvp/"
OUT="pixmmvp_output/" # Output predictions of all compared models
ANS="pixmmvp_answers/" # Jsonl answers of all compared models
META="pixmmvp_meta/" # Metadata output (txt) used for spacy score evaluation and (jsonl) automatic evaluation
VIS="pixmmvp_visualizations/" # Output visualizations

# Run 3 protocols
bash eval_omgllava.sh $DATA $OUT 0 $ANS
bash eval_omgllava.sh $DATA $OUT 1 $ANS
bash eval_omgllava.sh $DATA $OUT 3 $ANS

# Evaluate accuracies
echo "OMG-LLava"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate OMGLLAVACONDA
python ../eval/protocol1_accuracy.py --openai_api_key API_KEY --answer_file "$ANS/answers_omgllava_protocol1.jsonl"
python ../eval/protocol2_accuracy.py --answers_file "$ANS/answers_omgllava_protocol2.jsonl"

# Evaluate IoUs
python ../eval/eval_iou.py --dataset_root $DATA --preds_dir "$OUT/preds_omgllava_protocol1/"
python ../eval/eval_iou.py --dataset_root $DATA --preds_dir "$OUT/preds_omgllava_protocol3/"

###################################################### Vanilla MLLMs (not pixel-level) ##########################################################
# Llava 1.5 7B Example
# Set the following variables, and modify conda to your environment setup and API_KEY to OpenAI API Key.
# Similar setup to Cambrian (eval_cambrian.sh) .

# Run 3 protocols
bash eval_llava.sh $DATA $OUT 0 $ANS $META
bash eval_llava.sh $DATA $OUT 1 $ANS $META
bash eval_llava.sh $DATA $OUT 3 $ANS $META

# Evaluate accuracies
echo "LLava"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda deactivate
conda activate LLAVACONDA
python ../eval/protocol1_accuracy.py --openai_api_key OPENAI_API_KEY --answer_file "$ANS/answers_llava-1.5-7b-liu_protocol1.jsonl"
python ../eval/protocol2_accuracy.py --answers_file "$ANS/answers_llava-1.5-7b-liu_protocol2.jsonl"


# Evaluate IoUs
#Oracle evaluation
bash eval_pixfoundation.sh llava-1.5-7b-liu $DATA $OUT $VIS 0 $META oracle
bash eval_pixfoundation.sh llava-1.5-7b-liu $DATA $OUT $VIS 1 $META oracle

# a+s evaluation
bash eval_pixfoundation.sh llava-1.5-7b-liu $DATA $OUT $VIS 0 $META spacy_score
bash eval_pixfoundation.sh llava-1.5-7b-liu $DATA $OUT $VIS 1 $META spacy_score


# Auto selection + auto evaluation
bash infer_gptauto.sh llava-1.5-7b-liu $DATA $OUT 0 $META
bash infer_gptauto.sh llava-1.5-7b-liu $DATA $OUT 1 $META

bash eval_pixfoundation.sh llava-1.5-7b-liu $DATA $OUT $VIS 0 $META auto
bash eval_pixfoundation.sh llava-1.5-7b-liu $DATA $OUT $VIS 1 $META auto
