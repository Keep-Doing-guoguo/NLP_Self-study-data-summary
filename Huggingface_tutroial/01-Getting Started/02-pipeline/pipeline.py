from transformers.pipelines import SUPPORTED_TASKS

print(SUPPORTED_TASKS.items())

for k, v in SUPPORTED_TASKS.items():
    print(k, v)


from transformers import pipeline,AutoModelForSequenceClassification,AutoTokenizer,QuestionAnsweringPipeline,BertModel

#pipline得创建和使用方式
pipe = pipeline('text-classification')

print(pipe("very good!"))

#指定任务类型，再指定模型，创建基于指定模型得pipeline

# https://huggingface.co/models
pipe = pipeline("text-classification", model="uer/roberta-base-finetuned-dianping-chinese")

print(pipe("我觉得不太行！"))


# 这种方式，必须同时指定model和tokenizer
model1 = BertModel.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
model = AutoModelForSequenceClassification.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")#文本分类就是一个二分类问题
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(pipe("我觉得不太行！"))

print(pipe.model.device)


#CPU时间测试
# import torch
# import time
# times = []
# for i in range(100):
#     torch.cuda.synchronize()
#     start = time.time()
#     pipe("我觉得不太行！")
#     torch.cuda.synchronize()
#     end = time.time()
#     times.append(end - start)
# print(sum(times) / 100)

qa_pipe = pipeline("question-answering", model="uer/roberta-base-chinese-extractive-qa")
print(qa_pipe)
print(QuestionAnsweringPipeline)
'''

BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(21128, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)

'''
#这是一个文本分类任务，所以这个out_features就是2，也就是说是一个二分类任务。
'''
vocab_size = 21128
token_type_ids 指的是一个输入是否是两个句子。

'''