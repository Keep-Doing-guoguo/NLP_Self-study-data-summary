'''
多输出值处理
下面的例子展示了输出多个值时，以列表形式表现的处理方式。
'''
# import gradio as gr
# with gr.Blocks() as demo:
#     food_box = gr.Number(value=10, label="Food Count")
#     status_box = gr.Textbox()#这里应该是获取到Textbox的值
#     def eat(food):
#         if food > 0:
#             return food - 1, "full"
#         else:
#             return 0, "hungry"
#     gr.Button("EAT").click(
#         fn=eat,
#         inputs=food_box,
#         #根据返回值改变输入组件和输出组件
#         outputs=[food_box, status_box]
#     )
# demo.launch()

'''
下面的例子展示了输出多个值时，以字典形式表现的处理方式。
组件配置修改
事件监听器函数的返回值通常是相应的输出组件的更新值。
有时我们也想更新组件的配置，比如说可见性。
在这种情况下，我们可以通过返回update函数更新组件的配置。
'''
import gradio as gr
def change_textbox(choice):
    #根据不同输入对输出控件进行更新
    if choice == "short":
        return gr.update(lines=2, visible=True, value="Short story: ")
    elif choice == "long":
        return gr.update(lines=8, visible=True, value="Long story...")
    else:
        return gr.update(visible=False)
with gr.Blocks() as demo:
    radio = gr.Radio(
        ["short", "long", "none"], label="Essay Length to Write?"
    )
    text = gr.Textbox(lines=2, interactive=True)
    radio.change(fn=change_textbox, inputs=radio, outputs=text)
demo.launch()
#
