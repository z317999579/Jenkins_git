# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Decode from trained T2T models.

This binary performs inference using the Estimator API.

Example usage to decode from dataset:

  t2t-decoder \
      --data_dir ~/data \
      --problems=algorithmic_identity_binary40 \
      --model=transformer
      --hparams_set=transformer_base

Set FLAGS.decode_interactive or FLAGS.decode_from_file for alternative decode
sources.
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
import numpy as np
import cmd



### 使用自己定义的transformer.py

import tensorflow as tf
import operator,time

flags = tf.flags
FLAGS = flags.FLAGS
#flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
# Additional flags in bin/t2t_trainer.py and utils/flags.py




if __name__ == "__main__":
  ########## read test

  ################# init model
  import os
  #rootpath=os.environ['ROOT_PATH']
  #fpath=os.path.dirname(os.path.realpath(__file__))
  #f1=os.getcwd()
  rootpath= os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

  print rootpath

  data_dir = rootpath+'/data'
  problems = 'symptom_asking_problem'
  model = 'transformer'
  hparams_set = 'transformer_small'
  t2t_usr_dir = rootpath+'/src'
  output_dir = rootpath+'/model'

  return_beam=True
  beam_size=20
  write_beam_scores=False
  ###
  from ProblemDecoder_t2t144_RawSess import ProblemDecoder_predict as ProblemDecoder
  pd = ProblemDecoder(problem=problems,
                      model_dir=output_dir,
                      model_name=model,
                      hparams_set=hparams_set,
                      usr_dir=t2t_usr_dir,
                      data_dir=data_dir,
                      return_beams=return_beam,
                      write_beam_scores=write_beam_scores,
                      batch_size_specify=2,
                      beam_size=beam_size,
                      eos_required=False,
                      extra_length=10)


  sex = ''
  while(sex != '男' and sex != '女'):
    sex = raw_input('Input sex:').strip()

  #(,1] (1,3] (3,6] (6,9] (9,12] (12,18] (18,30] (30,40] (40,50] (50,60] (60,70] (70,75] (75,)
  ageSet = set(['(,1]', '(1,3]', '(3,6]', '(6,9]', '(9,12]', '(12,18]', '(18,30]', '(30,40]', '(40,50]', '(50,60]', '(60,70]', '(70,75]', '(75,)'])

  print(ageSet)

  age = ''
  while(not age in ageSet):
    age = raw_input('Input age:')


  print('{0} {1}'.format(sex,age))

  dict_file = data_dir+"/dict/symptom.dict.txt"

  dictSet = set(line.strip() for line in open(dict_file))

  sym = ''
  currentSym = [sex,age]
  while(sym != 'quit'):
    sym = raw_input('symtom:')
    if not sym in dictSet:
      continue
    currentSym.append(sym)
    retll = pd.infer_singleSample(' '.join(currentSym),10)
    for ret in retll:
      print("---------------------------")
      print ret

'''
    for alternativeInps in inps:
        print(alternativeInps)
        retll = pd.decode_from_list([alternativeInps])

        strill=[]

        for ret in retll:
            print("---------------------------")
            if return_beam == True and write_beam_scores == False:
                for beam in ret:
                    print beam
            elif return_beam == True and write_beam_scores == True:
                for score_beam in ret:
                    print("score is: {0}, result is [{1}]".format(score_beam[0], score_beam[1]))
            elif return_beam == False:
                print ret
                ret_stri=''.join(ret)
                strill.append(ret_stri)
            # d = json.loads(jstri)
            # dll.append(d)
    print ('done')

'''


