# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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

"""Data generators for PTB data-sets."""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import collections
import os
import sys
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import tensorflow as tf


EOS = text_encoder.EOS
flags = tf.flags
FLAGS = flags.FLAGS


cpath=FLAGS.data_dir
#cpath = u'/Users/sj/PycharmProjects/tensor2tensor/symptom_asking/data'


wordDict = cpath+u"/dict/symptom.dict.txt"
trainCorpus = cpath+u"/corpus/train.txt"
testCorpus = cpath+u"/corpus/test.txt"

@registry.register_problem("symptom_asking_problem")
class Symptom_Asking_Problem(text_problems.Text2SelfProblem):



  def is_generate_per_split(self):
    return True
  
  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def vocab_filename(self):
    return wordDict

  @property
  def vocab_type(self):
    return text_problems.VocabType.TOKEN

  def generate_samples(self, data_dir, tmp_dir, dataset_split):

    train_file, valid_file = trainCorpus, testCorpus

    assert train_file, "Training file not found"
    assert valid_file, "Validation file not found"

    train = dataset_split == problem.DatasetSplit.TRAIN
    filepath = train_file if train else valid_file

    with tf.gfile.GFile(filepath, "r") as f:
      for line in f:
        yield {"targets": line}

