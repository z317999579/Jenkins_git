#!/bin/bash
ORIGINAL_PATH=`pwd`
SHELL_PATH=`dirname $0`
cd "$SHELL_PATH/../"
ROOT_PATH=`pwd`
MODEL_PATH=$ROOT_PATH/model
#SRC_PATH=$ORIGINAL_PATH
SRC_PATH=$ROOT_PATH/src
DATA_PATH=$ROOT_PATH/data
DICT_PATH=$DATA_PATH/dict
CORPUS_PATH=$DATA_PATH/corpus
TRAINING_PATH=$DATA_PATH/training

echo $MODEL_PATH
echo $SRC_PATH
echo $DATA_PATH
echo $DICT_PATH
echo $CORPUS_PATH
echo $TRAINING_PATH
cd "$ORIGINAL_PATH"

MODEL=transformer
PROBLEM=symptom_asking_problem
PARAMSET=transformer_small
USR_DIR=$SRC_PATH
hparams="pos=none"
