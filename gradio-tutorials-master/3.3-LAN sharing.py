'''
通过设置server_name=‘0.0.0.0’（表示使用本机ip）,
server_port（可不改，默认值是7860）。
那么可以通过本机ip:端口号在局域网内分享应用。
#show_error为True表示在控制台显示错误信息。
demo.launch(server_name='0.0.0.0', server_port=8080, show_error=True)

这里host地址可以自行在电脑查询，
C:\Windows\System32\drivers\etc\hosts 修改一下即可 127.0.0.1再制定端口号
'''
