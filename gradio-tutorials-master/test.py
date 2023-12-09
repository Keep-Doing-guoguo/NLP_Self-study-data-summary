import os
from typing import List

import gradio as gr

print('debug')


def init_vector_store(file_obj):
    print('debug')
    print(file_obj)
    print(file_obj.name)


    return None


if __name__ == "__main__":
    block = gr.Blocks()
    with block as demo:

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""<h1><center>This is a Test Script</center></h1>
                        """)
                file = gr.File(label='请上传知识库文件',
                               file_types=['.txt', '.md', '.docx', '.pdf'])

                use_web = gr.Radio(["True", "False"],
                                   label="Web Search",
                                   value="False")

                init_vs = gr.Button("知识库文件向量化")




                output = gr.Textbox(label="Output Box")
            init_vs.click(
                init_vector_store,
                show_progress=True,
                inputs=[file],
                outputs=[output],
            )


    demo.queue(concurrency_count=3) \
        .launch(server_name='127.0.0.1', # ip for listening, 0.0.0.0 for every inbound traffic, 127.0.0.1 for local inbound
                server_port=7861, # the port for listening
                show_api=False, # if display the api document
                share=False, # if register a public url
                inbrowser=False) # if browser would be open automatically
