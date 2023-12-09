'''
组件渲染：点击作为输入
在某些情况下，您可能希望在实际在UI中呈现组件之前定义组件。
例如，您可能希望在相应的gr.Textbox输入上方显示使用gr.examples的示例部分。
由于gr.Examples需要输入组件对象作为参数，
因此您需要先定义输入组件，然后在定义gr.Exmples对象后再进行渲染。
解决方法是在gr.Blocks()范围外定义gr.Textbox，
并在UI中希望放置的任何位置使用组件的.render()方法。
'''
import gradio as gr
input_textbox = gr.Textbox()
with gr.Blocks() as demo:
    #提供示例输入给input_textbox，示例输入以嵌套列表形式设置
    gr.Examples(["hello", "bonjour", "merhaba"], input_textbox)
    # render函数渲染input_textbox
    input_textbox.render()
demo.launch()
