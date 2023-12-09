import gradio as gr
import numpy as np
import time
#生成steps张图片，每隔1秒钟返回
def fake_diffusion(steps):
    for _ in range(steps):
        time.sleep(1)
        image = np.random.randint(255, size=(300, 600, 3))#随机在255里面选择一个数字，后面是设置维度大小
        yield image
demo = gr.Interface(fake_diffusion,
                    #设置滑窗，最小值为1，最大值为10，初始值为3，每次改动增减1位
                    inputs=gr.Slider(1, 10, value=3, step=1),
                    outputs="image")
#生成器必须要queue函数
demo.queue()
demo.launch()