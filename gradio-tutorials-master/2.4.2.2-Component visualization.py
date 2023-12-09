'''
组件可视化：输出可视化从无到有
如下所示，我们可以通过visible和update函数构建更为复杂的应用。
'''

import gradio as gr
with gr.Blocks() as demo:
    # 出错提示框
    error_box = gr.Textbox(label="Error", visible=False)
    # 输入框
    name_box = gr.Textbox(label="Name")
    age_box = gr.Number(label="Age")
    symptoms_box = gr.CheckboxGroup(["Cough", "Fever", "Runny Nose"])
    submit_btn = gr.Button("Submit")
    # 输出不可见
    with gr.Column(visible=False) as output_col:
        diagnosis_box = gr.Textbox(label="Diagnosis")
        patient_summary_box = gr.Textbox(label="Patient Summary")
    def submit(name, age, symptoms):
        if len(name) == 0:
            return {error_box: gr.update(value="Enter name", visible=True)}
        if age < 0 or age > 200:
            return {error_box: gr.update(value="Enter valid age", visible=True)}
        return {
            output_col: gr.update(visible=True),
            diagnosis_box: "covid" if "Cough" in symptoms else "flu",
            patient_summary_box: f"{name}, {age} y/o"
        }
    submit_btn.click(
        submit,
        [name_box, age_box, symptoms_box],
        [error_box, diagnosis_box, patient_summary_box, output_col],
    )
demo.launch()
