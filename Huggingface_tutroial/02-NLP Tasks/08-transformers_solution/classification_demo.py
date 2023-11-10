# 文本分类实例 这些都是基于全部参数进行训练的。
## Step1 导入相关包
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
## Step2 加载数据集
dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)#过滤使用
a = dataset['review'][:20]
a1 = dataset['label'][:20]
## Step3 划分数据集
datasets = dataset.train_test_split(test_size=0.1)

## Step4 数据集预处理
import torch
path = 'C:/Users/16403/PycharmProjects/pythonProject/Huggingface_Toturials-main/transformers-code-master/02-NLP Tasks/08-transformers_solution/chinese-macbert-base'
tokenizer = AutoTokenizer.from_pretrained(path)#tokenizer的加载，词和数字的对应进行相加

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=32, truncation=True, padding="max_length")
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
a2 = tokenized_datasets['train']['input_ids'][:20]
a3 = tokenized_datasets['train']['token_type_ids'][:20]
a4 = tokenized_datasets['train']['attention_mask'][:20]

## Step5 创建模型
model = AutoModelForSequenceClassification.from_pretrained(path)#序列对序列模型，既有encoder又有decoder部分。
## Step6 创建评估函数
import evaluate

acc_metric = evaluate.load("accuracy")
f1_metirc = evaluate.load("f1")
def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metirc.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc
## Step7 创建TrainingArguments
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=1,   # 训练时的batch_size
                               gradient_accumulation_steps=32,  # *** 梯度累加 ***
                               gradient_checkpointing=True,     # *** 梯度检查点 ***
                               optim="adafactor",               # *** adafactor优化器 ***
                               per_device_eval_batch_size=1,    # 验证时的batch_size
                               num_train_epochs=1,              # 训练轮数
                               logging_steps=10,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型
train_args
## Step8 创建Trainer
from transformers import DataCollatorWithPadding

# *** 参数冻结 ***
for name, param in model.bert.named_parameters():
    param.requires_grad = False

trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)
## Step9 模型训练
trainer.train()
trainer.evaluate(tokenized_datasets["test"])
trainer.predict(tokenized_datasets["test"])
## Step10 模型预测
sen = "我觉得这家酒店不错，饭很好吃！"
id2_label = {0: "差评！", 1: "好评！"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sen, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=-1)
    print(f"输入：{sen}\n模型预测结果:{id2_label.get(pred.item())}")
from transformers import pipeline

model.config.id2label = id2_label
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
pipe(sen)
