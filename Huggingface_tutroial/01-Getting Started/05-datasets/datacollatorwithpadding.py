from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer

# 初始化一个分词器（tokenizer）
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 数据样本，通常是文本序列
data = [
    "This is an example sentence.",
    "Another example for padding.",
    "Short sentence."
]
data = ['我的名字叫顺溜！','我明年就要毕业了！','今天的天气真不错啊！但是早上的温度还是蛮低的。']

# 使用分词器对数据进行分词，并进行填充（padding）
# max_length 可以设置成适当的最大序列长度
data_collator = DataCollatorWithPadding(tokenizer, max_length=32)

# 将数据传递给数据收集器进行处理
input_data = tokenizer(data, padding=True, truncation=True, return_tensors="pt")#这里默认填充是25的吧？
a0 = tokenizer.decode(input_data['input_ids'][0])
a1 = tokenizer.decode(input_data['input_ids'][1])
a2 = tokenizer.decode(input_data['input_ids'][2])
output_data = data_collator(input_data)#并没有发现经过这里有任何变化
b0 = tokenizer.decode(output_data['input_ids'][0])
b1 = tokenizer.decode(output_data['input_ids'][1])
b2 = tokenizer.decode(output_data['input_ids'][2])

print(output_data)
print()
