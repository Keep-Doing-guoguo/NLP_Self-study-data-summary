import gradio as gr
def postprocess(message):
    # 在这里添加你的处理逻辑
    processed_message = message.upper()
    return processed_message


chatbot = gr.Chatbot()
chatbot.postprocess = postprocess

response = chatbot.send_message("你好")
print(response)  # 输出：你好
