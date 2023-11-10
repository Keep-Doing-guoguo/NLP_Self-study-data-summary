# 模型加载与保存
from transformers import AutoConfig, AutoModel, AutoTokenizer
## 在线加载
#model = AutoModel.from_pretrained("hfl/rbt3", force_download=True)#这个是原始的Bert模型结构
## 模型下载
# !git clone "https://huggingface.co/hfl/rbt3"
# !git lfs clone "https://huggingface.co/hfl/rbt3" --include="*.bin"
## 离线加载
path = 'C:/Users/16403/PycharmProjects/pythonProject/Huggingface_Toturials-main/transformers-code-master/01-Getting Started/04-model/hflrbt3'
model = AutoModel.from_pretrained(path)
## 模型加载参数
model = AutoModel.from_pretrained(path)
b = model.config
'''
BertConfig {
  "_name_or_path": "C:/Users/16403/PycharmProjects/pythonProject/Huggingface_Toturials-main/transformers-code-master/01-Getting Started/04-model/hflrbt3",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 3,
  "output_past": true,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

'''
config = AutoConfig.from_pretrained(path)
'''
BertConfig {
  "_name_or_path": "C:/Users/16403/PycharmProjects/pythonProject/Huggingface_Toturials-main/transformers-code-master/01-Getting Started/04-model/hflrbt3",
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 3,
  "output_past": true,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.33.2",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}
'''
a = config.output_attentions
from transformers import BertConfig
# 模型调用
sen = "弱小的我也有大梦想！"
tokenizer = AutoTokenizer.from_pretrained(path)
inputs = tokenizer(sen, return_tensors="pt")

## 不带Model Head的模型调用
model = AutoModel.from_pretrained(path, output_attentions=True)
output = model(**inputs)

a1 = output.last_hidden_state.size()
len(inputs["input_ids"][0])
## 带Model Head的模型调用
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
clz_model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=10)
a2 = clz_model(**inputs)
a3 = clz_model.config.num_labels
