from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.layers import common_hparams

@registry.register_hparams
def transformer_parsing_base_my():
  """Hparams for parsing on wsj only."""
  hparams = transformer.transformer_base()
  hparams.attention_dropout = 0.2
  #hparams.layer_prepostprocess_dropout = 0.2
  hparams.max_length = 512
  #hparams.learning_rate_warmup_steps = 16000
  hparams.hidden_size = 1024
  hparams.learning_rate = 0.05
  #hparams.shared_embedding_and_softmax_weights = int(False)
  return hparams




@registry.register_hparams
def lstm_attention_my():
  """hparams for LSTM with attention."""
  hparams = common_hparams.basic_params1()
  hparams.batch_size = 512
  hparams.hidden_size = 128
  hparams.num_hidden_layers = 2
  hparams.max_length = 100
  hparams.dropout=0.8
  hparams.learning_rate = 0.001

  # Attention
  hparams.add_hparam("attn_vec_size", hparams.hidden_size)
  return hparams


@registry.register_hparams
def transformer_base_single_gpu_my():
  """HParams for transformer base model for single gpu."""
  hparams = transformer.transformer_base()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 16000
  #hparams.hidden_size = 256
  hparams.learning_rate = 0.001
  hparams.num_decoder_layers= 3
  return hparams





