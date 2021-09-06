import os
import sqlite3


class MyDBManager:
    def __init__(self):
        base_dir = os.getcwd()
        self._db = base_dir + "/db/db_gpt2.db"


    def __enter__(self):
        self.conn = sqlite3.connect(self._db)
        return self.conn


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()
        if exc_val:
            raise Exception("MyDBManager error!")


connect = MyDBManager()


class DBConnect:
    def __init__(self, connect=connect):
        self.connect = connect


    def check_connect(self, chat_id):
        self.chat_id = chat_id
        with self.connect as conn:
            cursor = conn.cursor()
            row_query = """SELECT * FROM users where user_id=?"""
            value_query = (self.chat_id, )
            result = cursor.execute(row_query, value_query)
            records = result.fetchone()

            if records is None or len(records) == 0:
                if self.create_user(cursor, self.chat_id):
                    conn.commit()
                    return True
                else:
                    return False
        return True


    def create_user(self, cursor, chat_id):
        row_query = """INSERT INTO users ('user_id') VALUES (?)"""
        value_query = (chat_id, )
        cursor.execute(row_query, value_query)
        return True