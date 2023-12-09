'''
相比Interface，Blocks提供了一个低级别的API，用于设计具有更灵活布局和数据流的网络应用。
Blocks允许控制组件在页面上出现的位置，处理复杂的数据流（例如，输出可以作为其他函数的输入），并根据用户交互更新组件的属性可见性。

'''
import gradio as gr
def greet(name):
    return "Hello " + name + "!"
# with gr.Blocks() as demo:
#     #设置输入组件
#     name = gr.Textbox(label="Name")
#     # 设置输出组件
#     output = gr.Textbox(label="Output Box")
#     #设置按钮
#     greet_btn = gr.Button("Greet")
#     #设置按钮点击事件
#     greet_btn.click(fn=greet, inputs=name, outputs=output)

#查看row和column得关系。
with gr.Blocks() as demo:
    #设置输入组件
    name = gr.Textbox(label="Name")
    # 设置输出组件
    output = gr.Textbox(label="Output Box")
    #设置按钮
    greet_btn = gr.Button("Greet")
    #设置按钮点击事件
    greet_btn.click(fn=greet, inputs=name, outputs=output)
demo.launch()
'''
Blocks方式需要with语句添加组件，
如果不设置布局方式，那么组件将按照创建的顺序垂直出现在应用程序中，运行界面
'''