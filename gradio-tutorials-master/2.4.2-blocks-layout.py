'''

2.4.2 Blocks布局
Blocks应用的是html中的flexbox模型布局，默认情况下组件垂直排列。

组件水平排列
使用Row函数会将组件按照水平排列，但是在Row函数块里面的组件都会保持同等高度。
'''
#按照行进行布局显示。
import gradio as gr
with gr.Blocks() as demo:
    with gr.Row():
        img1 = gr.Image()
        text1 = gr.Text()
    btn1 = gr.Button("button")
demo.launch()