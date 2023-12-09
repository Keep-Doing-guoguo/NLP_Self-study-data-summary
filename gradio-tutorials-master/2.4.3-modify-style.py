'''
自定义css
要获得额外的样式功能，您可以设置行内css属性将任何样式给应用程序。如下所示。
'''
import gradio as gr
#修改blocks的背景颜色
with gr.Blocks(css=".gradio-container {background-color: red}") as demo:
    box1 = gr.Textbox(value="Good Job")
    box2 = gr.Textbox(value="Failure")
demo.launch()

'''
元素选择
您可以向任何组件添加HTML元素。通过elem_id选择对应的css元素。
'''
import gradio as gr
# 这里用的是id属性设置
with gr.Blocks(css="#warning {background-color: red}") as demo:
    box1 = gr.Textbox(value="Good Job", elem_id="warning")
    box2 = gr.Textbox(value="Failure")
    box3 = gr.Textbox(value="None", elem_id="warning")
demo.launch()