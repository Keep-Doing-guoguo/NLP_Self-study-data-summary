import gradio as gr

demo = gr.Blocks(css="""#btn {color: red} .abc {font-family: "Comic Sans MS", "Comic Sans", cursive !important}""")

with demo:
    default_json = {"a": "a"}

    num = gr.State(value=0)
    squared = gr.Number(value=0)
    btn = gr.Button("Next Square", elem_id="btn", elem_classes=["abc", "def"])

    stats = gr.State(value=default_json)
    table = gr.JSON()


    def increase(var, stats_history):
        var += 1
        stats_history[str(var)] = var ** 2
        return var, var ** 2, stats_history, stats_history


    btn.click(increase, [num, stats], [num, squared, stats, table])
    #它表示一个按钮点击事件。当用户点击按钮时，会执行increase函数，并将num和stats作为参数传递给该函数。同时，num、squared、stats和table也会作为参数传递给increase函数。
'''

increase 是在按钮点击时要执行的函数。

[num, stats] 是一个包含 increase 函数的参数的列表。这表示当按钮被点击时，会将 num 和 stats 这两个对象传递给 increase 函数。
[num, squared, stats, table] 则是指定了 Gradio 库将更新的变量。这表示 increase 函数的返回值将更新这些变量，以便在界面上反映出来。

具体来说，当按钮被点击时，Gradio 会调用 increase 函数，并将 num 和 stats 作为参数传递。
然后，increase 函数会修改这两个参数，并返回一个包含更新后的值的元组。Gradio 将根据指定的变量列表 [num, squared, stats, table] 更新相应的界面元素。
这种机制使得按钮点击事件与界面元素的交互非常方便，能够实时地更新和显示相关的信息。

'''
if __name__ == "__main__":
    print('de')
    demo.launch()
    print('de')