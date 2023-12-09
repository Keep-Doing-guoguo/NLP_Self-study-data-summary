'''
在首次打开网页前，可以设置账户密码。
比如auth参数为（账户，密码）的元组数据。
这种模式下不能够使用queue函数。
demo.launch(auth=("admin", "pass1234"))
如果想设置更为复杂的账户密码和密码提示，可以通过函数设置校验规则。

#账户和密码相同就可以通过
def same_auth(username, password):
    return username == password
demo.launch(auth=same_auth,auth_message="username and password must be the same")

'''