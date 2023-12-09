import gradio as gr
import numpy as np
def flip(im):
    return np.flipud(im)
demo = gr.Interface(
    flip,
    gr.Image(source="webcam", streaming=True),
    "image",
    live=True
)
demo.launch()
'''
在许多情形下，我们的输入是实时视频流或者音频流，
那么意味这数据不停地发送到后端，这是可以采用streaming模式处理数据。
'''