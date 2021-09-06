from .db_connect import DBConnect, connect


class DBVgg(DBConnect):
    def __init__(self, connect=connect):
        self.connect = connect
        self.style_params_list = ["None", "None", 20, 5, 0, 512]
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


    def add_base_params(self, chat_id):
        tmp_param_list = self.style_params_list.copy()
        tmp_param_list.insert(0, chat_id)

        with connect as conn:
            cursor = conn.cursor()
            row_query = """SELECT * FROM vgg_base_params where user_id=?"""
            value_query = (chat_id,)
            result = cursor.execute(row_query, value_query)
            records = result.fetchone()
            print(records)

            if records is None or len(records) == 0:
                row_query = """INSERT INTO vgg_base_params('user_id', 'content_link', 'style_link', 'param_epoch_num',
                                                            'param_show_every', 'param_device', 'param_image_size') 
                                                            VALUES (?, ?, ?, ?, ?, ?, ?)"""
                value_query = (*tmp_param_list,)

                cursor.execute(row_query, value_query)
                conn.commit()
        return True


    def add_base_layers_params(self, chat_id, base_style, base_content):
        style_params, content_params = base_style, base_content
        style_params.insert(0, chat_id)
        content_params.insert(0, chat_id)

        with connect as conn:
            cursor = conn.cursor()
            row_query = """SELECT * FROM content_layers where user_id=?"""
            value_query = (chat_id,)
            result = cursor.execute(row_query, value_query)
            records = result.fetchone()

            if records is None or len(records) == 0:
                row_query = """INSERT INTO content_layers ('user_id', 'conv1_1(3, 64)', 'conv1_2(64, 64)', 
                'conv2_1(64,128)', 'conv2_2(128,128)', 'conv3_1(128,256)', 'conv3_2(256,256)', 'conv3_3(256,256)', 
                'conv3_4(256,256)', 'conv4_1(256,512)', 'conv4_2(512,512)', 'conv4_3(512,512)', 'conv4_4(512,512)', 
                'conv5_1(512,512)', 'conv5_2(512,512)', 'conv5_3(512,512)', 'conv5_4(512,512)') 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

                value_query = (*content_params,)
                cursor.execute(row_query, value_query)

                row_query = """INSERT INTO style_layers ('user_id', 'conv1_1(3, 64)', 'conv1_2(64, 64)', 
                'conv2_1(64,128)', 'conv2_2(128,128)', 'conv3_1(128,256)', 'conv3_2(256,256)', 'conv3_3(256,256)', 
                'conv3_4(256,256)', 'conv4_1(256,512)', 'conv4_2(512,512)', 'conv4_3(512,512)', 'conv4_4(512,512)', 
                'conv5_1(512,512)', 'conv5_2(512,512)', 'conv5_3(512,512)', 'conv5_4(512,512)') 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

                value_query = (*style_params,)
                cursor.execute(row_query, value_query)

                conn.commit()

        return True


    def save_file_link(self, chat_id, file_path, is_style=False):
        try:
            if is_style:
                with connect as conn:
                    cursor = conn.cursor()
                    row_query = """UPDATE vgg_base_params SET style_link=? WHERE user_id=?"""
                    value_query = (file_path, chat_id,)

                    cursor.execute(row_query, value_query)
                    conn.commit()
            else:
                with connect as conn:
                    cursor = conn.cursor()
                    row_query = """UPDATE vgg_base_params SET content_link=? WHERE user_id=?"""
                    value_query = (file_path, chat_id,)

                    cursor.execute(row_query, value_query)
                    conn.commit()
            return True
        except Exception as ex:
            print(f"{ex}: Download error content-style image.")
            return False


    def get_base_vgg_params(self, chat_id):
        base_params = {}
        try:
            with connect as conn:
                cursor = conn.cursor()

                row_query = """SELECT * FROM vgg_base_params WHERE user_id=?"""
                value_query = (chat_id, )
                records = cursor.execute(row_query, value_query).fetchone()

                print(records)

                base_params['epoch number'] = records[4]
                base_params['show cost [steps]'] = records[5]
                base_params['device[0:cpu, 1:cuda]'] = records[6]
                base_params['image size'] = records[7]

            return base_params
        except Exception as ex:
            print(f"{ex}: VGG params transfer error")
            return False


    def save_vgg_base_params(self, param):
        param[0] = param[0].replace('epoch number', 'param_epoch_num').replace('show cost [steps]', 'param_show_every')\
                .replace('device[0:cpu, 1:cuda]', 'param_device').replace('image size', 'param_image_size')
        param[1] = int(param[1])
        param[2] = int(param[2])

        #Доп заглушка, все равно cuda нет.
        if param[0] == 'device[0:cpu, 1:cuda]':
            return True

        with connect as conn:
            cursor = conn.cursor()
            row_query = f"""UPDATE vgg_base_params SET {param[0]}=? WHERE user_id=?"""
            value_query = (param[1], param[2])#Да, это жесть! Туплю
            cursor.execute(row_query, value_query)
            conn.commit()
        return True


    def get_style_params(self, chat_id, style_layer=True):
        with connect as conn:
            cursor = conn.cursor()
            table_layer = 'style_layers' if style_layer else 'content_layers'
            row_query = f"""SELECT * FROM {table_layer} where user_id=?"""
            value_query = (chat_id,)

            result = cursor.execute(row_query, value_query)
            records = result.fetchone()

            if len(records) > 0:
                layers_params = [int(x) for x in records[2:]]
            else:
                print("No data!")
                return None
        return layers_params


    def save_layer_param(self, param, style_layer=True):
        try:
            param[1] = float(bool(int(param[1])))
            param[2] = int(param[2])
        except Exception as ex:
            return False

        with connect as conn:
            cursor = conn.cursor()
            table_layer = 'style_layers' if style_layer else 'content_layers'
            row_query = f"""UPDATE {table_layer} SET '{param[0]}'=? WHERE user_id=?"""
            value_query = (param[1], param[2])#Да, это жесть! Туплю
            cursor.execute(row_query, value_query)
            conn.commit()
        return True


    def reset_layers(self, chat_id, base_style, base_content):
        style_params, content_params = base_style, base_content
        style_params.append(chat_id)
        content_params.append(chat_id)

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


    def load_file_link(self, chat_id, is_style=False):
        with connect as conn:
            cursor = conn.cursor()

            if is_style:
                row_query = """SELECT style_link FROM vgg_base_params WHERE user_id=?"""
            else:
                row_query = """SELECT content_link FROM vgg_base_params WHERE user_id=?"""

            value_query = (chat_id,)
            result = cursor.execute(row_query, value_query)
            records = result.fetchone()

            if len(records[0]) > 10:
                return records[0]

        return None