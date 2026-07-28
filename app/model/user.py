from app.core.database import db


class User:
    @staticmethod
    def get_all():
        with db.get_cursor() as cursor:
            cursor.execute('SELECT * FROM users ORDER BY id DESC')
            return cursor.fetchall()

    @staticmethod
    def find_by_username(username):
        with db.get_cursor() as cursor:
            cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
            return cursor.fetchone()

    @staticmethod
    def find_by_id(user_id: int):
        with db.get_cursor() as cursor:
            cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
            return cursor.fetchone()

    """新增用户"""
    @staticmethod
    def create(username:str, password_hash:str, role:str="user"):
        with db.get_cursor() as cursor:
            cursor.execute('INSERT INTO users (username, password_hash,role) VALUES (%s, %s,%s)',
                           (username, password_hash,role))
            #用于获取最后一次插入操作生成的自动递增 ID
            return cursor.lastrowid

    @staticmethod
    def update_username(user_id: int, username: str):
        # 按主键更新比按旧用户名更新更稳，避免用户已改名后条件失效。
        with db.get_cursor() as cursor:
            cursor.execute(
                'UPDATE users SET username = %s WHERE id = %s',
                (username, user_id),
            )
            return cursor.rowcount

    @staticmethod
    def update_password(user_id: int, password_hash: str):
        with db.get_cursor() as cursor:
            cursor.execute(
                'UPDATE users SET password_hash = %s WHERE id = %s',
                (password_hash, user_id),
            )
            return cursor.rowcount
