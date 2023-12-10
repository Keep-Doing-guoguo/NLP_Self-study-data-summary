l = [1,3,5,7]
it =iter(l)
print(next(it))
print(next(it))
print(next(it))
print(next(it))
#print(next(it))StopIteration

'''
 迭代器的用途:
        用迭代器可以依次访问可迭代对象的数据
'''
l = [1,3,5,7]
it = iter(l)
while True:
    try:
        x=next(it)
        print(x)
    except StopIteration:
        break



#训练测试

s={'唐僧','悟空','悟能','悟净'}
it = iter(s)
try:
    while True:
        x = next(it)
        print(x)
except StopIteration:
    print('遍历结束')