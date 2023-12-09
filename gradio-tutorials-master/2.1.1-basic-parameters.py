#2.基本参数-支持的接口
#2.1 Interface类以及基础模块
'''
Gradio 可以包装几乎任何 Python 函数为易于使用的用户界面。从上面例子我们看到，简单的基于文本的函数。但这个函数还可以处理很多类型。 Interface类通过以下三个参数进行初始化：

fn：包装的函数
inputs：输入组件类型，（例如：“text”、"image）
ouputs：输出组件类型，（例如：“text”、"image）
通过这三个参数，我们可以快速创建一个接口并发布他们。

最常用的基础模块构成。

应用界面：gr.Interface(简易场景), gr.Blocks(定制化场景)

输入输出：gr.Image(图像), gr.Textbox(文本框), gr.DataFrame(数据框), gr.Dropdown(下拉选项), gr.Number(数字), gr.Markdown, gr.Files

控制组件：gr.Button(按钮)

布局组件：gr.Tab(标签页), gr.Row(行布局), gr.Column(列布局)

'''
#1.2.1自定义组件
# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!"
# demo = gr.Interface(
#     fn=greet,
#     # 自定义输入框
#     # 具体设置方法查看官方文档
#     inputs=gr.Textbox(lines=3, placeholder="Name Here...",label="my input"),
#     outputs="text",
# )
# demo.launch()


#1.2.2多个输入和输出
import gradio as gr
#该函数有3个输入参数和2个输出参数
# def greet(name, is_morning, temperature):
#     salutation = "Good morning" if is_morning else "Good evening"
#     greeting = f"{salutation} {name}. It is {temperature} degrees today"
#     celsius = (temperature - 32) * 5 / 9
#     return greeting, round(celsius, 2)
# demo = gr.Interface(
#     fn=greet,
#     #按照处理程序设置输入组件
#     inputs=["text", "checkbox", gr.Slider(0, 100)],
#     #按照处理程序设置输出组件
#     outputs=["text", "number"],
# )
# demo.launch()



#1.2.3图像组件
import numpy as np
import gradio as gr
def sepia(input_img):
    #处理图像
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img
#shape设置输入图像大小
demo = gr.Interface(sepia, gr.Image(), "image")
demo.launch()