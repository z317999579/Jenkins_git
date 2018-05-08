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
import json
import threading
import logging,logging.handlers

#### log output
fileRotator = logging.handlers.RotatingFileHandler('./service-nlp-nfyy.log',maxBytes=1024*100)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
fileRotator.setFormatter(formatter)
logging.getLogger("nlp").addHandler(fileRotator)
logging.getLogger("nlp").setLevel(logging.INFO)

### 使用自己定义的transformer.py

import tensorflow as tf
import operator,time

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
# Additional flags in bin/t2t_trainer.py and utils/flags.py




if __name__ == "__main__":
    ########## read test
    inps = ['男 (70,75] 头痛 恶心 呕吐 头晕']

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

    #FLAGS.decode_interactive = True
    #FLAGS.decode_from_file='./input_string.txt'
    #FLAGS.decode_to_file='./result.txt'

    #main()

    return_beam=True
    beam_size=10
    write_beam_scores=True
    ###
    from ProblemDecoder_t2t144 import ProblemDecoder
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






    retll = pd.decode_from_list(inps)

    strill=[]

    for ret in retll:
        if return_beam == True and write_beam_scores == False:
            for beam in ret:
                print beam
        elif return_beam == True and write_beam_scores == True:
            for score_beam in ret:
                print score_beam[0], score_beam[1]
        elif return_beam == False:
            print ret
            ret_stri=''.join(ret)
            strill.append(ret_stri)
        # d = json.loads(jstri)
        # dll.append(d)
    print ('done')

    #####
    inpll,outll=[],[]
    for ii in range(len(inps)):
        inp,out=inps[ii],strill[ii]
        inpll.append(inp)
        outll.append(out)

    print ''
    
    print(inp)
    print(out)




