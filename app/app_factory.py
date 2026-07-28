"""
创建FastApi应用，统一挂载路由
"""
from fastapi import FastAPI

from app.core.config import Settings
from app.core.exceptions import register_exception_handlers
from app.routers.health import router as health_router
from app.routers.db_test import router as db_test_router
from app.routers.users import router as users_router
from app.routers.auth import router as auth_router
from app.routers.devices import router as devices_router
from app.routers.ocr import router as ocr_router


def create_app():
    app = FastAPI(title=Settings.APP_NAME)
    # 通过 include_router 注册后，实际访问的完整路径是/api/health
    app.include_router(health_router, prefix="/api")
    app.include_router(db_test_router, prefix="/api")
    app.include_router(users_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(devices_router, prefix="/api")
    # OCR 使用与现有业务接口相同的 /api 前缀和登录认证方式。
    app.include_router(ocr_router, prefix="/api")

    register_exception_handlers(app)
    return app
