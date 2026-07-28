
from ..core.auth import TokenStore, extract_bearer_token
from ..core.responses import success_response
from ..model.login import verify_password
from ..model.user import User
from fastapi import APIRouter, Depends,Header
from ..core.security import hash_password
from ..schems.auth import PasswordChangeRequest, ProfileUpdateRequest, RegisterRequest
from ..dependencies.auth import current_user
from ..core.exceptions import AppException

router = APIRouter()
"""
注册用户
payload: RegisterRequest  自动验证请求体，符合模型则通过
"""


@router.post("/auth/register")
def register(payload: RegisterRequest):
    existing = User.find_by_username(payload.username)
    if existing:
        raise AppException(400,"用户名已存在")
    user_id = User.create(
        payload.username,
        hash_password(payload.password),
        payload.role
    )
    # return {"success": True, "user_id": user_id}
    return success_response(
        message="注册成功",
        user_id=user_id,
    )

"""登录"""


@router.post("/auth/login")
def login(payload: RegisterRequest):
    user = User.find_by_username(payload.username)
    if not user:
        raise AppException(401,"用户名或密码错误")
    if not verify_password(payload.password, user["password_hash"]):
        raise AppException(401,"用户名或密码错误")

    token = TokenStore.create_token(user)

    # return {"success": True, "message": "登陆成功",
    #         "access_token": token,  # 新增token
    #         "token_type": "bearer",  # token类型
    #         "user": {
    #             "id": user["id"],
    #             "username": user["username"],
    #             "role": user["role"],
    #             "created_at": user["created_at"],
    #         }, }
    return success_response(
        message="登录成功",
        data={
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"],
                "created_at": user["created_at"],
            },
        },
    )


@router.get("/auth/me")
def me(user=Depends(current_user)):
    """
    获取当前登录用户信息。
    只有带着合法 token 才能访问。
    """
    return success_response(data=user)


@router.put("/auth/profile")
def update_profile(
    payload: ProfileUpdateRequest,
    authorization: str | None = Header(None),
    user=Depends(current_user),
):
    # 编辑资料当前只支持用户名，先把能力做实，不为未来字段先扩模型。
    existing = User.find_by_username(payload.username)
    if existing and existing["id"] != user["id"]:
        raise AppException(400, "用户名已存在")

    User.update_username(user["id"], payload.username)
    refreshed_user = User.find_by_id(user["id"])
    if not refreshed_user:
        raise AppException(404, "用户不存在")

    token = extract_bearer_token(authorization)
    TokenStore.update_user(token, refreshed_user)
    return success_response(
        message="资料更新成功",
        data={
            "id": refreshed_user["id"],
            "username": refreshed_user["username"],
            "role": refreshed_user["role"],
        },
    )


@router.post("/auth/change-password")
def change_password(payload: PasswordChangeRequest, user=Depends(current_user)):
    db_user = User.find_by_id(user["id"])
    if not db_user:
        raise AppException(404, "用户不存在")
    if not verify_password(payload.current_password, db_user["password_hash"]):
        raise AppException(400, "当前密码错误")

    User.update_password(user["id"], hash_password(payload.new_password))
    return success_response(message="密码修改成功")

"""
Header(default=None)：从请求头里拿 Authorization
extract_bearer_token(...)：把 Bearer xxx 里的 xxx 取出来
TokenStore.revoke_token(...)：删除这个 token，让它失效
"""
@router.post("/auth/logout")
def logout(authorization: str | None = Header(None)):
    print(f"authorization:{authorization}")
    # token = TokenStore.create_token(authorization)
    token = extract_bearer_token(authorization)
    print(f"token被移除:{token}")
    TokenStore.revoke_token(token)
    # return {"success": True, "message": "退出成功"}
    return success_response(message="退出成功")