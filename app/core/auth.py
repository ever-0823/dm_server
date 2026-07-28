import secrets

from fastapi import Header
from ..core.exceptions import AppException
from typing import Optional


class TokenStore:
    _tokens = {}

    @classmethod
    def create_token(cls, user: dict) -> str:
        """"""
        # 用于生成安全随机 URL 安全令牌
        token = secrets.token_urlsafe(32)
        # print(f"token: {token}")
        cls._tokens[token] = {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
        }
        print(cls._tokens)
        return token

    # Optional[dict]返回值：可能返回 dict 或 None
    @classmethod
    def get_user(cls, token: str) -> Optional[dict]:
        """
        根据token获取用户
        :param token:
        :return:
        """
        return cls._tokens.get(token)

    @classmethod
    def revoke_token(cls, token: str):
        """让token失效"""
        cls._tokens.pop(token, None)

    @classmethod
    def update_user(cls, token: str, user: dict):
        # ponytail: 现有登录态存在内存里，改资料后顺手同步这一份就够了。
        if token not in cls._tokens:
            return
        cls._tokens[token] = {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
        }

# 用于从 HTTP 请求的 Authorization 头中提取 Bearer Token
def extract_bearer_token(authorization: Optional[str]) -> str:
    """
    从请求头authorization中提取bearer token
    :return:
    """
    # 检查 Authorization 头是否存在
    if not authorization:
        raise AppException(401, "请先登录")
    # 标准 Bearer Token 前缀 Bearer eyJhbGciOiJIUzI1NiIs...
    prefix = "Bearer"
    # 检查格式是否正确
    if not authorization.startswith(prefix):
        raise AppException(401, "认证信息格式错误")
    return authorization[len(prefix):].strip()
def get_current_user(authorization: Optional[str] = Header(default=None)):
    """
    根据请求头里的token获取当前登录用户
    如果token失效，就返回401
    :return:
    """
    token = extract_bearer_token(authorization)
    user = TokenStore.get_user(token)
    if not user:
        raise AppException(401, "登录已失效，请重新登录")
    return user
