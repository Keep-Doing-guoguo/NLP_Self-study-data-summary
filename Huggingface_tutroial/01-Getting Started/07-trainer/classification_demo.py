from torch.optim import optimizer, Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import torch
import evaluate


dataset = load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train")
dataset = dataset.filter(lambda x: x["review"] is not None)#去掉None值。
datasets = dataset.train_test_split(test_size=0.1)#划分训练集和测试集
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

def process_function(examples):#相当于是转换为tokenized
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)

model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
trainset, validset = tokenized_datasets["train"], tokenized_datasets["test"]
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))




acc_metric = evaluate.load("accuracy")
f1_metirc = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metirc.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc
#这个玩意就是用来加载参数的，到下面共下面使用的。
train_args = TrainingArguments(output_dir="./checkpoints",      # 输出文件夹
                               per_device_train_batch_size=64,  # 训练时的batch_size
                               per_device_eval_batch_size=128,  # 验证时的batch_size
                               logging_steps=10,                # log 打印的频率
                               evaluation_strategy="epoch",     # 评估策略
                               save_strategy="epoch",           # 保存策略
                               save_total_limit=3,              # 最大保存数
                               learning_rate=2e-5,              # 学习率
                               weight_decay=0.01,               # weight_decay
                               metric_for_best_model="f1",      # 设定评估指标
                               load_best_model_at_end=True)     # 训练完成后加载最优模型

clf_metrics = evaluate.combine(["accuracy", "f1", "recall", "precision"])#将会计算三种

trainer = Trainer(model=model,
                  args=train_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric)

trainer.train()

trainer.evaluate(tokenized_datasets["test"])

trainer.predict(tokenized_datasets["test"])
optimizer = Adam(model.parameters(), lr=2e-5)

def evaluate():
    model.eval()
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            clf_metrics.add_batch(predictions=pred.long(), references=batch["labels"].long())
    return clf_metrics.compute()
#下面是自己写的一个训练函数和上面的评测函数，进行一个对比。
def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        clf = evaluate()
        print(f"ep: {ep}, {clf}")


train()

#自己进行模型测试。
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















