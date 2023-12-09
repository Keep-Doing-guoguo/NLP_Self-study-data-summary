'''
3.1 互联网分享
如果运行环境能够连接互联网，在launch函数中设置share参数为True，
那么运行程序后。
Gradio的服务器会提供XXXXX.gradio.app地址。
通过其他设备，比如手机或者笔记本电脑，都可以访问该应用。
这种方式下该链接只是本地服务器的代理，
不会存储通过本地应用程序发送的任何数据。
这个链接在有效期内是免费的，
好处就是不需要自己搭建服务器，坏处就是太慢了，
毕竟数据经过别人的服务器。
demo.launch(share=True)#设置
'''
'''
3.2 huggingface托管
为了便于向合作伙伴永久展示我们的模型App,
可以将gradio的模型部署到 HuggingFace的 Space托管空间中，
完全免费的哦。

方法如下：

1，注册huggingface账号：https://huggingface.co/join

2，在space空间中创建项目：https://huggingface.co/spaces

3，创建好的项目有一个Readme文档，可以根据说明操作，也可以手工编辑app.py和requirements.txt文件。
'''