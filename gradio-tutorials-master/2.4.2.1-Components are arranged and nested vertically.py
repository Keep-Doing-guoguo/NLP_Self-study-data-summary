'''
组件垂直排列与嵌套
组件通常是垂直排列，我们可以通过Row函数和Column函数生成不同复杂的布局。
'''
import gradio as gr
with gr.Blocks() as demo:
    with gr.Row():#设置第一行
        text1 = gr.Textbox(label="t1")#这些并没有按照行来进行划分，所以默认按照列显示。
        slider2 = gr.Textbox(label="s2")
        drop3 = gr.Dropdown(["a", "b", "c"], label="d3")
    with gr.Row():#然后设置第二行
        # scale与相邻列相比的相对宽度。例如，如果列A的比例为2，列B的比例为1，则A的宽度将是B的两倍。
        # min_width设置最小宽度，防止列太窄
        with gr.Column(scale=2, min_width=600):#在行里面设置列。
            text1 = gr.Textbox(label="prompt 1")
            text2 = gr.Textbox(label="prompt 2")
            inbtw = gr.Button("Between")
            text4 = gr.Textbox(label="prompt 1")
            text5 = gr.Textbox(label="prompt 2")
        with gr.Column(scale=1, min_width=600):
            # img1 = gr.Image("test.jpg")
            btn = gr.Button("Go")
            btn1 = gr.Button("Go1")
demo.launch()