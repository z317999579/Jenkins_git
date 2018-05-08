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










if __name__ == "__main__":
    import os
    #rootpath=os.environ['ROOT_PATH']
    rootpath='../'
    print rootpath

    # FLAGS.data_dir = rootpath + '/data'
    # FLAGS.problems = 'symptom_asking_problem'
    # FLAGS.model = 'transformer'
    # FLAGS.hparams_set = 'transformer_small'
    # FLAGS.t2t_usr_dir = rootpath + '/src'
    # FLAGS.output_dir = rootpath + '/model'
    # hparams_key_value = "pos=none"
    # FLAGS.hparams="pos=none"

    data_dir = rootpath + '/data'
    problems = 'symptom_asking_problem'
    model = 'transformer'
    #model='transformer_scorer_yr'#

    hparams_set = 'transformer_small'
    t2t_usr_dir = rootpath + '/src'
    output_dir = rootpath + '/model'
    #hparams_key_value=None
    hparams_key_value = "pos=none"
    #hparams_key_value='eval_run_autoregressive=False'

    #FLAGS.decode_interactive = True
    #FLAGS.decode_from_file='./input_string.txt'
    #FLAGS.decode_to_file='./result.txt'

    #main()
    #inp_str = 'symptom @@ 乳房痛 //symptom'
    #inpll = ['symptom @@ 乳房痛 //symptom', 'symptom @@ 失眠 //symptom']




    ### predict mode
    import ProblemDecoder_predict as pp
    pd=pp.ProblemDecoder_predict(problem=problems,
                      model_dir=output_dir,
                      model_name=model,
                      hparams_set=hparams_set,
                      usr_dir=t2t_usr_dir,
                      data_dir=data_dir,
                      batch_size_specify=2,
                      return_beams=True,
                      beam_size=10,
                      write_beam_scores=True,
                      hparams_key_value=hparams_key_value
                      )

    # pd = ProblemDecoder(problem=FLAGS.problems,
    #                     model_dir=FLAGS.output_dir,
    #                     model_name=FLAGS.model,
    #                     hparams_set=FLAGS.hparams_set,
    #                     usr_dir=FLAGS.t2t_usr_dir,
    #                     data_dir=FLAGS.data_dir,
    #                     batch_size_specify=2,
    #                     return_beams=True,
    #                     beam_size=10,
    #                     write_beam_scores=True,
    #                     hparams_key_value=hparams_key_value
    #                     )

    #inp_str=['symptom @@ 乳房痛 //symptom','symptom @@ 失眠 //symptom']
    inp_str=['男 (12,18] 发热']
    #inps= [inp_str]
    stri,score=pd.infer_singleSample(inp_str[0],10) #class [1,1,vocab]

    for x in stri:
        print(x.decode(encoding='UTF-8',errors='strict'))
    print(score)

    #arr=pd.infer_batch_seq2seq(inp_str)









