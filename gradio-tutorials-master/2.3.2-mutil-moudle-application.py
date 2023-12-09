import numpy as np
import gradio as gr
def flip_text(x):#这里可以切换文本内容
    return x[::-1]#进行倒叙输出
def flip_image(x):#可以切换图片模块
    return np.fliplr(x)#数组反转
with gr.Blocks() as demo:
    #用markdown语法编辑输出一段话
    gr.Markdown("Flip text or image files using this demo.")
    # 设置tab选项卡
    with gr.Tab("Flip Text"):#切换选项
        #Blocks特有组件，设置所有子组件按垂直排列
        #垂直排列是默认情况，不加也没关系
        with gr.Column():
            text_input = gr.Textbox()
            text_output = gr.Textbox()
            text_button = gr.Button("Flip")
    with gr.Tab("Flip Image"):
        #Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")
    #设置折叠内容
    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")
    text_button.click(flip_text, inputs=text_input, outputs=text_output)#这里是设置button点击
    image_button.click(flip_image, inputs=image_input, outputs=image_output)
demo.launch()