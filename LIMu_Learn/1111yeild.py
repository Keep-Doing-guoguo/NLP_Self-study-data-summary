def foo():
    #print("starting...")
    while True:
        res = yield 4
        print("res:",res)
g = foo()
#返回一个4，然后再进行打印res，其为none
print(next(g))
print(next(g))
#print("*"*20)
