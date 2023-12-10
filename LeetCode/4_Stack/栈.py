#创建一个栈
stack = []
stack.append(1)
stack.append(2)
stack.append(3)
#[1,2,3]
print(stack)
print(stack[-1])
print(stack)

temp = stack.pop()
print(temp)
print(stack)
print(len(stack))
while len(stack) > 0:
    temp = stack.pop()
    print(temp)