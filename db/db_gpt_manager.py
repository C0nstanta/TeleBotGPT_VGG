from .db_connect import DBConnect, connect
import json

class DBGpt(DBConnect):
    def __init__(self, connect=connect):
        self.connect = connect
        self.base_params = [256, 3, 1, 100, 0.9, 0.6, 5, 0, 1, 1]
        self.gpt_params = {
                'max_length': 256,
                'no_repeat_ngram_size': 3,
                'do_sample': True,
                'top_k': 100,
                'top_p': 0.9,
                'temperature': 0.6,
                'num_return_sequences': 5,
                'device': 0,
                'is_always_use_length': True,
                'length_generate': '1',
            }


    def check_gpt_params(self, chat_id):
        with self.connect as conn:
            cursor = conn.cursor()
            row_query = """SELECT * FROM gpt3_params WHERE user_id=?"""
            value_query = (chat_id, )
            result = cursor.execute(row_query, value_query)
            records = result.fetchone()

            if records is None or len(records) == 0:
                if self.add_base_params(cursor, chat_id):
                    conn.commit()
                else:
                    return False
        return True


    def add_base_params(self, cursor, chat_id):
        args = self.base_params.copy()
        args.insert(0, chat_id)

        row_query = """INSERT INTO gpt3_params ('user_id', 'max_length', 'no_repeat_ngram_size', 'do_sample', 'top_k',
        'top_p', 'temperature', 'num_return_sequences', 'device', 'is_always_use_length', 'length_generate') 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        value_query = (*args, )
        cursor.execute(row_query, value_query)

        return True


    def get_params(self, chat_id):
        tmp_gpt_params = self.gpt_params.copy()
        with self.connect as conn:
            cursor = conn.cursor()

            row_query = """SELECT * FROM gpt3_params WHERE user_id=?"""
            value_query = (chat_id, )
            records = cursor.execute(row_query, value_query).fetchone()

            if records is None or len(records) == 0:
                return False
            else:
                tmp_gpt_params['max_length'] = records[2]
                tmp_gpt_params['no_repeat_ngram_size'] = records[3]
                tmp_gpt_params['do_sample'] = records[4]
                tmp_gpt_params['top_k'] = records[5]
                tmp_gpt_params['top_p'] = records[6]
                tmp_gpt_params['temperature'] = records[7]
                tmp_gpt_params['num_return_sequences'] = records[8]
                tmp_gpt_params['device'] = records[9]
                tmp_gpt_params['is_always_use_length'] = records[10]
                tmp_gpt_params['length_generate'] = records[11]

            return tmp_gpt_params


    def save_params(self, param):
        if param[0] == 'temperature' or param[0] == 'top_p':
            param[1] = float(param[1])
        else:
            param[1] = int(param[1])
        param[2] = int(param[2])

        with self.connect as conn:
            cursor = conn.cursor()
            row_query = f"""UPDATE gpt3_params SET {param[0]}=? WHERE user_id=?"""
            value_query = (param[1], param[2])#Да, это жесть! Туплю
            cursor.execute(row_query, value_query)
            conn.commit()
        return True


    def reset_params(self, chat_id):
        with self.connect as conn:
            cursor = conn.cursor()
            row_query = """UPDATE gpt3_params SET 'max_length'=?, 'no_repeat_ngram_size'=?, 'do_sample'=?,
                                                  'top_k'=?, 'top_p'=?, 'temperature'=?, 'num_return_sequences'=?, 
                                                  'device'=?, 'is_always_use_length'=?, 'length_generate'=? 
                                                  WHERE user_id=?"""
            value_query = (*self.base_params, chat_id)
            print(self.base_params)
            print("reset before")
            cursor.execute(row_query, value_query)
            conn.commit()
            print("reset after")

        return True


    def get_dialogue(self, chat_id):
        with self.connect as conn:
            cursor = conn.cursor()

            row_query = """SELECT user_dialogue FROM  gpt3_params WHERE user_id=?"""
            value_query = (chat_id, )
            result = cursor.execute(row_query, value_query)
            records = result.fetchone()

            print(records)

            if records[0] is not None and len(records[0]) > 0:
                return json.loads(records[0])
            else:
                return None


    def clear_dialogue(self, chat_id):
        with self.connect as conn:
            cursor = conn.cursor()

            row_query = """UPDATE gpt3_params SET user_dialogue=NULL WHERE user_id=?"""
            value_query = (chat_id,)
            cursor.execute(row_query, value_query)
            conn.commit()

        if self.get_dialogue(chat_id) == None:
            return True
        else:
            return False



    def rewrite_dialogue(self, chat_id, data):
        j_string = json.dumps(data, ensure_ascii=False).encode('utf8')
        with self.connect as conn:
            cursor = conn.cursor()

            row_query = """UPDATE gpt3_params SET user_dialogue=? WHERE user_id=?"""
            value_query = (j_string.decode(), chat_id)
            cursor.execute(row_query, value_query)
            conn.commit()
        return True