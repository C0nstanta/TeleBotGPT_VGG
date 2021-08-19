import os
import sqlite3
import json


BASE_PARAMS = {
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


class DBManager:
    def __init__(self, connect=connect):

        self.connect = connect
        self.base_params = [256, 3, 1, 100, 0.9, 0.6, 5, 0, 1, 1]
        self.user_params = {
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


    def check_connect(self, chat_id, args):
        self.chat_id = chat_id
        tmp_params = []
        tmp_params.append(chat_id)
        tmp_params.extend(self.base_params)

        with self.connect as conn:
            cursor = conn.cursor()
            row_query = """SELECT * FROM users where user_id=?"""
            value_query = (self.chat_id, )
            result = cursor.execute(row_query, value_query)
            records = result.fetchall()

            if len(records) == 0:
                self.create_user(cursor, args)
                self.add_base_params(cursor, tmp_params)

            conn.commit()
        return records


    def create_user(self, cursor, args):
        row_query = """INSERT INTO users ('user_id', 'user_fname', 'user_lname') VALUES (?, ?, ?)"""
        value_query = (*args, )
        cursor.execute(row_query, value_query)


    def add_base_params(self, cursor, args):
        row_query = """INSERT INTO user_params ('user_id', 'max_length', 'no_repeat_ngram_size', 'do_sample', 'top_k',
        'top_p', 'temperature', 'num_return_sequences', 'device', 'is_always_use_length', 'length_generate') 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        value_query = (*args,)
        cursor.execute(row_query, value_query)


    def add_style_transfer(self, cursor):
        row_query = """INSERT INTO style_transfer ('user_id', 'content_link', 'style_link') VALUES (?, ?, ?)"""
        value_query = (self.chat_id,'0', '1')
        cursor.execute(row_query, value_query)


    def get_params(self, user_id):
        with self.connect as conn:
            cursor = conn.cursor()

            row_query = """SELECT * FROM user_params WHERE user_id=?"""
            value_query = (user_id, )
            records = cursor.execute(row_query, value_query).fetchall()[0]

            self.user_params['max_length'] = records[2]
            self.user_params['no_repeat_ngram_size'] = records[3]
            self.user_params['do_sample'] = records[4]
            self.user_params['top_k'] = records[5]
            self.user_params['top_p'] = records[6]
            self.user_params['temperature'] = records[7]
            self.user_params['num_return_sequences'] = records[8]
            self.user_params['device'] = records[9]
            self.user_params['is_always_use_length'] = records[10]
            self.user_params['length_generate'] = records[11]

        return self.user_params


    def save_params(self, param):
        if param[0] == 'temperature' or param[0] == 'top_p':
            param[1] = float(param[1])
        else:
            param[1] = int(param[1])
        param[2] = int(param[2])

        with self.connect as conn:
            cursor = conn.cursor()
            row_query = f"""UPDATE user_params SET {param[0]}=? WHERE user_id=?"""
            value_query = (param[1],param[2])#Да, это жесть! Туплю
            cursor.execute(row_query, value_query)
            conn.commit()


    def get_dialogue(self):
        try:
            with self.connect as conn:
                cursor = conn.cursor()

                row_query = """SELECT * FROM  users WHERE user_id=?"""
                value_query = (self.chat_id,)
                result = cursor.execute(row_query, value_query)
                records = result.fetchmany(1)

                if records[0][4] is not None and len(records[0][4]) > 0:
                    return json.loads(records[0][4])
                else:
                    return None
        except Exception as ex:
            return None


    def rewrite_dialogue(self, data):
        j_string = json.dumps(data, ensure_ascii=False).encode('utf8')
        with self.connect as conn:
            cursor = conn.cursor()

            row_query = """UPDATE users SET user_dialog=? WHERE user_id=?"""
            value_query = (j_string.decode(), self.chat_id)
            cursor.execute(row_query, value_query)
            conn.commit()


    def clear_dialogue(self):
        with self.connect as conn:
            cursor = conn.cursor()

            row_query = """UPDATE users SET user_dialog=NULL WHERE user_id=?"""
            value_query = (self.chat_id, )
            cursor.execute(row_query, value_query)
            conn.commit()

        if self.get_dialogue() == None:
            return True
        else:
            return False


class DBStyle_Transfer(DBManager):
    def __init__(self, chat_id):
        self.chat_id = chat_id
        # self.style_params_list = [20, 5, 0, 512, [[0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [0, 1]]]
        self.style_params_list = ["None", "None", 20, 5, 0, 512, 1, 1, 1, 1, 10, 1]
        self.style_params = {
                'style_link': 'None',
                'content_link': 'None',
                'epoch_num': 20,
                'show_every': 5,
                'device': 0,
                'image_size': 512,
                'content_layer': [{"conv1_1": 0, "conv2_1": 0, "conv3_1": 0, "conv4_1": 0, "conv4_2": 1, "conv5_1": 0}],
                'style_layer': [{"conv1_1": 1, "conv2_1": 1, "conv3_1": 1, "conv4_1": 1, "conv4_2": 0, "conv5_1": 1}],
            }


    def add_base_params(self):
        param_list = self.style_params_list
        param_list.insert(0, self.chat_id)

        try:
            with connect as conn:
                cursor = conn.cursor()
                row_query = """SELECT * FROM style_transfer where user_id=?"""
                value_query = (self.chat_id,)
                result = cursor.execute(row_query, value_query)
                records = result.fetchmany(1)

                if len(records) == 0:
                    row_query = """INSERT INTO style_transfer ('user_id', 'content_link', 'style_link', 'param_epoch_num', 
                    'param_show_every', 'param_device', 'param_image_size', 'param_conv1_1', 'param_conv2_1', 'param_conv3_1', 
                    'param_conv4_1', 'param_conv4_2', 'param_conv5_1') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                    value_query = (*param_list,)

                    cursor.execute(row_query, value_query)
                    conn.commit()

            return True
        except Exception as ex:
            return False


    def add_base_layers_params(self, base_style, base_content):
        style_params, content_params = base_style, base_content
        style_params.insert(0, self.chat_id)
        content_params.insert(0, self.chat_id)
        try:
            with connect as conn:
                cursor = conn.cursor()
                row_query = """SELECT * FROM content_layers where user_id=?"""
                value_query = (self.chat_id,)
                result = cursor.execute(row_query, value_query)
                records = result.fetchmany(1)

                if len(records) == 0:
                    row_query = """INSERT INTO content_layers ('user_id', 'conv1_1(3, 64)', 'conv1_2(64, 64)', 'conv2_1(64,128)',
                    'conv2_2(128,128)', 'conv3_1(128,256)', 'conv3_2(256,256)', 'conv3_3(256,256)', 'conv3_4(256,256)', 'conv4_1(256,512)',
                    'conv4_2(512,512)', 'conv4_3(512,512)', 'conv4_4(512,512)', 'conv5_1(512,512)', 'conv5_2(512,512)',
                    'conv5_3(512,512)', 'conv5_4(512,512)') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                    value_query = (*content_params,)
                    cursor.execute(row_query, value_query)

                    row_query = """INSERT INTO style_layers ('user_id', 'conv1_1(3, 64)', 'conv1_2(64, 64)', 'conv2_1(64,128)', 
                    'conv2_2(128,128)', 'conv3_1(128,256)', 'conv3_2(256,256)', 'conv3_3(256,256)', 'conv3_4(256,256)', 'conv4_1(256,512)', 
                    'conv4_2(512,512)', 'conv4_3(512,512)', 'conv4_4(512,512)', 'conv5_1(512,512)', 'conv5_2(512,512)', 
                    'conv5_3(512,512)', 'conv5_4(512,512)') VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

                    value_query = (*style_params,)
                    cursor.execute(row_query, value_query)

                    conn.commit()

            return True
        except Exception as ex:
            return False


    def get_cl_values(self):
        with connect as conn:
            try:
                cursor = conn.cursor()
                row_query = """SELECT * FROM content_layers where user_id=?"""
                value_query = (self.chat_id,)
                result = cursor.execute(row_query, value_query)
                records = result.fetchmany(1)

                if len(records) > 0:
                    cl_list = [int(x) for x in records[0][2:]]
                    return cl_list
                else:
                    print("No data!")
                    return False
            except Exception as ex:
                return False


    def get_sl_values(self):
        with connect as conn:
            try:
                cursor = conn.cursor()
                row_query = """SELECT * FROM style_layers where user_id=?"""
                value_query = (self.chat_id,)
                result = cursor.execute(row_query, value_query)
                records = result.fetchmany(1)

                if len(records) > 0:
                    sl_list = [int(x) for x in records[0][2:]]
                    return sl_list
                else:
                    print("No data!")
                    return False
            except Exception as ex:
                return False


    def get_base_vgg_params(self, user_id):
        model_vgg = {}
        try:
            with connect as conn:
                cursor = conn.cursor()

                row_query = """SELECT * FROM style_transfer WHERE user_id=?"""
                value_query = (user_id, )
                records = cursor.execute(row_query, value_query).fetchmany(1)

                model_vgg['epoch numbers'] = records[0][4]
                model_vgg['show cost [steps]'] = records[0][5]
                model_vgg['device[0:cpu, 1:cuda]'] = records[0][12]
                model_vgg['image size'] = records[0][13]

            return model_vgg
        except Exception as ex:
            return False


    def get_base_vgg_layers(self, user_id):
        try:
            with connect as conn:
                cursor = conn.cursor()
                row_query = """SELECT * FROM style_transfer WHERE user_id=?"""
                value_query = (user_id, )
                result = cursor.execute(row_query, value_query).fetchmany(1)

        except Exception as ex:
            return False

    def save_file_link(self, file_path, is_style=False):
        try:
            if is_style:
                with connect as conn:
                    cursor = conn.cursor()
                    row_query = """UPDATE style_transfer SET style_link=? WHERE user_id=?"""
                    value_query = (file_path, self.chat_id,)

                    cursor.execute(row_query, value_query)
                    conn.commit()
            else:
                with connect as conn:
                    cursor = conn.cursor()
                    row_query = """UPDATE style_transfer SET content_link=? WHERE user_id=?"""
                    value_query = (file_path, self.chat_id,)

                    cursor.execute(row_query, value_query)
                    conn.commit()
            return True
        except Exception as ex:
            return False


    def load_file_link(self, is_style=False):
        try:
            if is_style:
                with connect as conn:
                    cursor = conn.cursor()
                    row_query = """SELECT style_link FROM style_transfer WHERE user_id=?"""
                    value_query = (self.chat_id,)

                    result = cursor.execute(row_query, value_query)
                    records = result.fetchmany(1)

                    if len(records[0][0]) > 10:
                        return records[0][0]
                    else:
                        return False
            else:
                with connect as conn:
                    cursor = conn.cursor()
                    row_query = """SELECT content_link FROM style_transfer WHERE user_id=?"""
                    value_query = (self.chat_id,)

                    result = cursor.execute(row_query, value_query)
                    records = result.fetchmany(1)

                    if len(records[0][0]) > 10:
                        return records[0][0]
                    else:
                        return False

        except Exception as ex:
            return False


    def save_style_params(self, param):
        param[0] = param[0].replace('epoch numbers', 'param_epoch_num').replace('show cost [steps]', 'param_show_every')\
                .replace('device[0:cpu, 1:cuda]', 'param_device').replace('image size', 'param_image_size')
        param[1] = int(param[1])
        param[2] = int(param[2])

        #Доп заглушка, все равно cuda нет.
        if param[0] == 'device[0:cpu, 1:cuda]':
            return True

        with connect as conn:
            cursor = conn.cursor()
            row_query = f"""UPDATE style_transfer SET {param[0]}=? WHERE user_id=?"""
            value_query = (param[1], param[2])#Да, это жесть! Туплю
            cursor.execute(row_query, value_query)
            conn.commit()
        return True


    def save_cl_params(self, param):
        param[1] = float(bool(int(param[1])))
        param[2] = int(param[2])

        with connect as conn:
            cursor = conn.cursor()
            row_query = f"""UPDATE content_layers SET '{param[0]}'=? WHERE user_id=?"""
            value_query = (param[1], param[2])#Да, это жесть! Туплю
            cursor.execute(row_query, value_query)
            conn.commit()
        return True


    def save_sl_params(self, param):
        param[1] = float(bool(int(param[1])))
        param[2] = int(param[2])

        with connect as conn:
            cursor = conn.cursor()
            row_query = f"""UPDATE style_layers SET '{param[0]}'=? WHERE user_id=?"""
            value_query = (param[1], param[2])#Да, это жесть! Туплю
            cursor.execute(row_query, value_query)
            conn.commit()
        return True



    def reset_layers(self, base_style, base_content):
        style_params, content_params = base_style, base_content
        style_params.append(self.chat_id)
        content_params.append(self.chat_id)
        try:
            with connect as conn:
                cursor = conn.cursor()
                row_query = """UPDATE  content_layers SET 'conv1_1(3, 64)'=?, 'conv1_2(64, 64)'=?, 'conv2_1(64,128)'=?,
                'conv2_2(128,128)'=?, 'conv3_1(128,256)'=?, 'conv3_2(256,256)'=?, 'conv3_3(256,256)'=?,
                'conv3_4(256,256)'=?, 'conv4_1(256,512)'=?, 'conv4_2(512,512)'=?, 'conv4_3(512,512)'=?,
                'conv4_4(512,512)'=?, 'conv5_1(512,512)'=?, 'conv5_2(512,512)'=?, 'conv5_3(512,512)'=?,
                'conv5_4(512,512)'=?  WHERE user_id=?"""
                value_query = (*content_params,)
                cursor.execute(row_query, value_query)

                row_query = """UPDATE  style_layers SET 'conv1_1(3, 64)'=?, 'conv1_2(64, 64)'=?, 'conv2_1(64,128)'=?,
                'conv2_2(128,128)'=?, 'conv3_1(128,256)'=?, 'conv3_2(256,256)'=?, 'conv3_3(256,256)'=?,
                'conv3_4(256,256)'=?, 'conv4_1(256,512)'=?, 'conv4_2(512,512)'=?, 'conv4_3(512,512)'=?,
                'conv4_4(512,512)'=?, 'conv5_1(512,512)'=?, 'conv5_2(512,512)'=?, 'conv5_3(512,512)'=?,
                'conv5_4(512,512)'=?  WHERE user_id=?"""
                value_query = (*style_params,)
                cursor.execute(row_query, value_query)

                conn.commit()

            return True
        except Exception as ex:
            return False