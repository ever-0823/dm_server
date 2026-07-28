from app.core.auth import TokenStore


def test_token_store_update_user() -> None:
    # 修改资料后，当前登录状态中的用户名也要同步更新。
    token = TokenStore.create_token({"id": 1, "username": "old_name", "role": "user"})
    TokenStore.update_user(token, {"id": 1, "username": "new_name", "role": "admin"})
    user = TokenStore.get_user(token)
    assert user == {"id": 1, "username": "new_name", "role": "admin"}
    TokenStore.revoke_token(token)


if __name__ == "__main__":
    test_token_store_update_user()
    print("ok")
