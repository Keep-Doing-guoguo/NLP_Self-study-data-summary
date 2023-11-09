import torch

x = torch.tensor([1, 2, 3])
print(x.repeat_interleave(2))
#tensor([1, 1, 2, 2, 3, 3])
# 传入多维张量，默认`展平`
y = torch.tensor([[1, 2], [3, 4]])
print(torch.repeat_interleave(y, 2))
#tensor([1, 1, 2, 2, 3, 3, 4, 4])
# 指定维度
print((y))
print(torch.repeat_interleave(y,3,0))
# tensor([[1, 2],
#         [1, 2],
#         [1, 2],
#         [3, 4],
#         [3, 4],
#         [3, 4]])
print(torch.repeat_interleave(y, 3, dim=1))
# tensor([[1, 1, 1, 2, 2, 2],
#         [3, 3, 3, 4, 4, 4]])
# 指定不同元素重复不同次数
print(torch.repeat_interleave(y, torch.tensor([1, 2]), dim=0))
# tensor([[1, 2],
#         [3, 4],
#         [3, 4]])
