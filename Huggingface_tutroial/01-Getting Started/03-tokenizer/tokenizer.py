from transformers import AutoTokenizer
sen = "弱小的我也有大梦想!"
# 从HuggingFace加载，输入模型名称，即可加载对于的分词器
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")


# tokenizer 保存到本地
#C:\Users\16403\PycharmProjects\pythonProject\Huggingface_Toturials-main\transformers-code-master\01-Getting Started\03-tokenizer\roberta_tokenizer
tokenizer.save_pretrained("./roberta_tokenizer")

# 从本地加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer/")#和上面得效果是相同的。

tokens = tokenizer.tokenize(sen)

a = tokenizer.vocab
a1 = tokenizer.vocab_size

# 将词序列转换为id序列
ids = tokenizer.convert_tokens_to_ids(tokens)

# 将id序列转换为token序列
tokens = tokenizer.convert_ids_to_tokens(ids)


# 将token序列转换为string
str_sen = tokenizer.convert_tokens_to_string(tokens)


# 将字符串转换为id序列，又称之为编码
ids1 = tokenizer.encode(sen, add_special_tokens=True)

# 将id序列转换为字符串，又称之为解码
str_sen = tokenizer.decode(ids, skip_special_tokens=False)

# 填充
ids2 = tokenizer.encode(sen, padding="max_length", max_length=15)


# 截断
ids3 = tokenizer.encode(sen, max_length=5, truncation=True)


ids4 = tokenizer.encode(sen, padding="max_length", max_length=15)


attention_mask = [1 if idx != 0 else 0 for idx in ids]
token_type_ids = [0] * len(ids)



inputs = tokenizer.encode_plus(sen, padding="max_length", max_length=15)


inputs = tokenizer(sen, padding="max_length", max_length=15)


sens = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]
res = tokenizer(sens)



# 单条循环处理
for i in range(1000):
    tokenizer(sen)



# 处理batch数据
res1 = tokenizer([sen] * 1000)


sen2 = "弱小的我也有大Dreaming!"

fast_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")


slow_tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese", use_fast=False)



# 单条循环处理
for i in range(10000):
    fast_tokenizer(sen)
print()