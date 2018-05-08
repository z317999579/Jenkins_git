python t2t-trainer_forPythonStart \
--hparams_set=transformer_base_single_gpu \
--problems=chunk_problem \
--model=transformer \
--train_steps=1000000 \
--batch_size=8192 \
--output_dir=/disk4/fushan_corpus/hadoop_test/problem_xianbingshi_chunks/template_xianbingshi_chunk_meituanyun/model_empty \
--model_dir=/disk4/fushan_corpus/hadoop_test/problem_xianbingshi_chunks/template_xianbingshi_chunk_meituanyun/model_empty \
--data_dir=/disk4/fushan_corpus/hadoop_test/problem_xianbingshi_chunks/template_xianbingshi_chunk_meituanyun/data


 

#t2t-trainer \
  #--t2t_usr_dir $USR_DIR \
  #--data_dir=$DATA_PATH \
  #--problems=$PROBLEM \
  #--model=$MODEL \
  #--hparams_set=$PARAMSET \
  #--output_dir=$MODEL_PATH \
  #--worker_gpu_memory_fraction 0.5 \
  #--save_checkpoints_secs 1200
  #--train_steps=6000

#--hparams_set=transformer_base_single_gpu --problems=zhusuChunk_problem --model=transformer --train_steps=250000 --batch_size=8192
