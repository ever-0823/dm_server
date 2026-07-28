from pydantic import BaseModel, Field, field_validator, model_validator

"""
这是一个使用 Pydantic 定义的数据验证模型，用于在 FastAPI 中接收和验证客户端发送的请求数据。
"""


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)  # 必填，必须是字符串类型
    password: str = Field(..., min_length=3, max_length=100)

    role: str = "user"  # 可选，默认值为 "user"

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("用户不能为空")
        return value

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("角色只能是user或admin")
        return value


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=3, max_length=100)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("用户不能为空")
        return value


class ProfileUpdateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        # 编辑资料目前只支持修改用户名，先把输入边界收紧到和注册一致。
        value = value.strip()
        if not value:
            raise ValueError("用户不能为空")
        return value


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=3, max_length=100)
    new_password: str = Field(..., min_length=3, max_length=100)

    @model_validator(mode="after")
    def validate_passwords(self):
        # 防止把新密码改成和旧密码一样，省掉一次无意义更新。
        if self.current_password == self.new_password:
            raise ValueError("新密码不能和当前密码相同")
        return self
