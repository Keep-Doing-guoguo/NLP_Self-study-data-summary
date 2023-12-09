import gradio as gr

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2

iface = gr.Interface(
    calculator,
    ["number", gr.inputs.Radio(["add", "subtract", "multiply", "divide"]), "number"],
    "number",
    live=True,
)#在Interface添加live=True参数，只要输入发生变化，结果马上发生改变。

iface.launch()
