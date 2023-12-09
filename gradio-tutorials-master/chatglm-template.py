import gradio as gr
def predict():
    print('predict')

def clear_session():
    print('clear_session')

if __name__ == "__main__":
    with gr.Blocks() as demo:#在这里设置一个总界面

        with gr.Row():#先设置为行分布

            with gr.Column(scale=4):#这是第二列得设置，下面为设置内容。
                chatbot = gr.Chatbot(label='ChatLLM').style(height=400)##设置一个对话窗，这是一个部分。

                message = gr.Textbox(label='请输入问题')#问题输入，这又是一个部分。
                state = gr.State()#状态对象可以用于存储和管理程序的状态信息。通过将状态对象赋值给变量state，可以在后续的代码中使用该状态对象来访问和修改状态信息。

                with gr.Row():#这又是一个部分，这部分为按钮部分。

                    clear_history = gr.Button("🧹 清除历史对话")

                    send = gr.Button("🚀 发送")
                    send.click(predict,
                               inputs=[ message,  state ],#状态得保存
                               outputs=[message, chatbot, state])
                    clear_history.click(fn=clear_session,
                                        inputs=[],
                                        outputs=[chatbot, state],
                                        queue=False)

                    message.submit(predict,
                                   inputs=[ message, state ],
                                   outputs=[ message, chatbot, state ])#这个和send.click得点击是相同得。


    demo.queue().launch(share=False)