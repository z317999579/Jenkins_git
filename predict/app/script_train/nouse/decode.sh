#!/bin/bash
source configuration.sh
echo 'train...'
echo $PROBLEM 
echo $USR_DIR
echo $MODEL
echo $PARAMSET
echo $MODEL_PATH
echo $DATA_PATH
cp transformer.py /usr/local/lib/python2.7/dist-packages/tensor2tensor/models/transformer.py 
python ProblemDecoder.py --hparams=$hparams
#t2t-trainer \
#  --t2t_usr_dir $USR_DIR \
#  --data_dir=$DATA_PATH \
#  --problems=$PROBLEM \
#  --model=$MODEL \
#  --hparams_set=$PARAMSET \
#  --output_dir=$MODEL_PATH \
#  --worker_gpu_memory_fraction 0.95 \
#  --save_checkpoints_secs 1200 \
#  --eval_steps 50000 
  #--train_steps=6000






