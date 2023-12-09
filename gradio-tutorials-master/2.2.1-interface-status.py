import gradio as gr
# scores = []
# def track_score(score):
#     scores.append(score)
#     #返回分数top3
#     top_scores = sorted(scores, reverse=True)[:3]
#     return top_scores
# demo = gr.Interface(
#     track_score,#这里放的即使函数名称
#     gr.Number(label="Score"),#输入端的名称显示
#     gr.JSON(label="Top Scores")#输出端的名称显示
# )
# demo.launch()


###dialect-status
import random
import gradio as gr
def chat(message, history):
    history = history or []
    message = message.lower()
    if message.startswith("how many"):
        response = random.randint(1, 10)
    elif message.startswith("how"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"
    history.append((message, response))
    return history, history
#设置一个对话窗
chatbot = gr.Chatbot().style(color_map=("green", "pink"))
demo = gr.Interface(
    chat,
    # 添加state组件
    ["text", "state"],#在你的函数中传入一个额外的参数，它代表界面的状态。：这行代码定义了输入组件，包括一个文本框和一个状态组件。文本框用于接收用户输入的文本，状态组件用于表示界面的状态。
    [chatbot, "state"],#这行代码定义了输出组件，包括一个聊天机器人和一个状态组件。聊天机器人用于显示处理后的结果，状态组件用于表示界面的状态。这个状态意味这，要返回得结果在聊天框上。
    # 设置没有保存数据的按钮
    allow_flagging="never",#这行代码设置不允许保存数据的标志为"never"，即不允许用户保存数据。
)
demo.launch()