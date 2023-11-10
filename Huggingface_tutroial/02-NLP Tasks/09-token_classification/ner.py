import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification,AutoModelForSequenceClassification

# 如果可以联网，直接使用load_dataset进行加载
#ner_datasets = load_dataset("peoples_daily_ner", cache_dir="./data")
# 如果无法联网，则使用下面的方式加载数据集

from datasets import DatasetDict
ner_datasets = DatasetDict.load_from_disk("ner_data")

'''
DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 20865
    })
    validation: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 2319
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags'],
        num_rows: 4637
    })
})
'''
#这是测试数据，查看数据结构信息。
aa = ner_datasets['train']
a = ner_datasets['train'][0]#这是在取出来具体的某一列元素的值。
b = ner_datasets['train'][1]
b1 = ner_datasets['train'][2]
a1 = ner_datasets['train'].features

#在文本处理中，token可以是一个词语、数字、标点符号、单个字母或任何可以成为文本分析的单个元素。
label_list = ner_datasets["train"].features["ner_tags"].feature.names#['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']对应的数字就是[0,1,2,3,4,5,6]
path = 'C:/Users/16403/PycharmProjects/pythonProject/Huggingface_Toturials-main/transformers-code-master/02-NLP Tasks/09-token_classification/chinese-macbert-base'
tokenizer = AutoTokenizer.from_pretrained(path)#加载模型


#这是测试数据，查看数据结构信息。
text = '我是中国人！'#这里面总共有6个字符，在前面加上一个CLS，再加一个SEP结尾，总共是6+2个字符。
# b2 = tokenizer(text, is_split_into_words=True)
b3 = tokenizer(text)
b4 = ner_datasets["train"][0]["tokens"]
b5 = tokenizer(b4, is_split_into_words=True)
b6 = tokenizer(b4)#因为传入的list中文字是单个的，所以这里产生的input_ids为单个的。
b7 = b5.word_ids()

#model = AutoModelForTokenClassification.from_pretrained(path)#加载这个模型，是为了查看模型得结构信息。

#这是测试数据，查看数据结构信息。
a2 = tokenizer(ner_datasets["train"][0]["tokens"], is_split_into_words=True)   # 对于已经做好tokenize的数据，要指定is_split_into_words参数为True
print()
a22 = tokenizer.decode(a2['input_ids'])#每一个句子有一个CLS 作为开头 SEP作为结束标志。
aa2 = ner_datasets["train"][0]["tokens"]
aa3 = tokenizer(ner_datasets["train"][0]["tokens"])
res = tokenizer("interesting word")#因为英文会将其拆分为一个字词。比如会将interesting是由4个数字来组合得。word由一个数字来组合。那么是如何知道得呢，需要调用word_ids()
a3 = res.word_ids()#[None, 0, 0, 0, 0, 1, None]这个显示的是前4个0代表一个单词，后一个1为一个单词。因为有可能有东西是字母。

# 借助word_ids 实现标签映射
def process_function(examples):
    tokenized_exmaples = tokenizer(examples["tokens"], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_exmaples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_exmaples["labels"] = labels
    return tokenized_exmaples
#tokenized_datasets = process_function(ner_datasets)
###################Test###################
# def process_function1(examples):
#
#     print('1')
#     print('1')
#     print('1')
#     tokenized_exmaples = tokenizer(examples["tokens"], max_length=128, truncation=True, is_split_into_words=True)
#     labels = []
#     for i, label in enumerate(examples["ner_tags"]):
#         word_ids = tokenized_exmaples.word_ids(batch_index=i)
#         label_ids = []
#         for word_id in word_ids:
#             if word_id is None:
#                 label_ids.append(-100)
#             else:
#                 label_ids.append(label[word_id])
#         labels.append(label_ids)
#     tokenized_exmaples["labels"] = labels
#     return tokenized_exmaples
###################Test###################
tokenized_datasets = ner_datasets.map(process_function, batched=True)#batched=True就是对整个batch的数据进行map，而不是一条一条的
a4 = tokenized_datasets["train"][0]#其中ner_tag代表得就是target。
a5 = tokenized_datasets['train']#总共数据显示由20865条数据

# 对于所有的非二分类任务，切记要指定num_labels，否则就会device错误，这是一个ner任务，所以并不是二分类任务。
model = AutoModelForTokenClassification.from_pretrained(path, num_labels=len(label_list))#(classifier): Linear(in_features=768, out_features=7, bias=True)

model1 = AutoModelForTokenClassification.from_pretrained(path)# (classifier): Linear(in_features=768, out_features=2, bias=True)

model2 = AutoModelForSequenceClassification.from_pretrained(path)
a6 = model.config.num_labels
# 这里方便大家加载，替换成了本地的加载方式，无需额外下载
seqeval = evaluate.load("seqeval_metric.py")#其实这里可以进行pip的


import numpy as np


def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)

    # 将id转换为原始的字符串类型的标签
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")

    return {
        "f1": result["overall_f1"]
    }
args = TrainingArguments(#配置训练参数
    output_dir="models_for_ner",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_steps=50,
    num_train_epochs=3#
)
#训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)
trainer.train()
trainer.evaluate(eval_dataset=tokenized_datasets["test"])

#STEP9
from transformers import pipeline
# 使用pipeline进行推理，要指定id2label
model.config.id2label = {idx: label for idx, label in enumerate(label_list)}
model.config

# 如果模型是基于GPU训练的，那么推理时要指定device
# 对于NER任务，可以指定aggregation_strategy为simple，得到具体的实体的结果，而不是token的结果
ner_pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")#在这里加载得是自己得模型和tokenizer.
res = ner_pipe("小明在北京上班")

# 根据start和end取实际的结果
ner_result = {}
x = "小明在北京上班"
for r in res:
    if r["entity_group"] not in ner_result:
        ner_result[r["entity_group"]] = []
    ner_result[r["entity_group"]].append(x[r["start"]: r["end"]])

ner_result





'''
模型结构：
vocabulary_size :21128
model_structure:

BertForTokenClassification(
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
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)

'''






































