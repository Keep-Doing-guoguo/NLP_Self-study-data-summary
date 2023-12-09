'''
可交互设置
任何输入的组件内容都是可编辑的，而输出组件默认是不能编辑的。
如果想要使得输出组件内容可编辑，设置interactive=True即可。

'''
# import gradio as gr
# def greet(name):
#     return "Hello " + name + "!"
# with gr.Blocks() as demo:
#     name = gr.Textbox(label="Name")
#     # 不可交互
#     # output = gr.Textbox(label="Output Box")
#     # 可交互
#     output = gr.Textbox(label="Output", interactive=True)
#     greet_btn = gr.Button("Greet")
#     greet_btn.click(fn=greet, inputs=name, outputs=output)
# demo.launch()

'''
事件设置
我们可以为不同的组件设置不同事件，如为输入组件添加change事件。
可以进一步查看官方文档，看看组件还有哪些事件
'''
# import gradio as gr
# def welcome(name):
#     return f"Welcome to Gradio, {name}!"
# with gr.Blocks() as demo:
#     gr.Markdown(
#     """
#     # Hello World!
#     Start typing below to see the output.
#     """)#这里是设置标题信息的
#     inp = gr.Textbox(placeholder="What is your name?")#设置隐形消息
#     out = gr.Textbox()
#     #设置change事件
#     inp.change(fn = welcome, inputs = inp, outputs = out)
# demo.launch()
'''
多个数据流
如果想处理多个数据流，只要设置相应的输入输出组件即可。
'''
import gradio as gr
def increase(num):
    return num + 1
with gr.Blocks() as demo:
    a = gr.Number(label="a")#设置一个number
    b = gr.Number(label="b")
    # 要想b>a，则使得b = a+1
    atob = gr.Button("b > a")
    atob.click(increase, a, b)
    # 要想a>b，则使得a = b+1
    btoa = gr.Button("a > b")
    btoa.click(increase, b, a)

demo.launch(share=True)