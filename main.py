# 导入 Uvicorn 服务器——这是一个基于 uvloop 的高性能 ASGI 服务器，用于运行 FastAPI 应用。它是 FastAPI 官方推荐的服务器之一
import os
import sys

import uvicorn
# app = FastAPI()
from app.app_factory import create_app

# 调用工厂函数，生成 FastAPI 应用实例，赋值给 app 变量
app = create_app()


def configure_utf8_console() -> None:
    """让后端日志和重载子进程统一使用 UTF-8 输出。"""
    # Uvicorn reload 会创建子进程，环境变量确保子进程也不会退回 GBK。
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}
"""
app 要运行的实例
host 监听地址
reload=true 开发模式：代码变动时自动重启服务器
"""
if __name__ == "__main__":
    configure_utf8_console()
    # 使用导入字符串启动，才能让 reload=True 在开发环境里正确监听代码变更。
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
