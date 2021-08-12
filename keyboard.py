from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove


START_KB = [['Just Talk', 'Params'],
            ['Clear Talk Data', 'Exit'],
            ['Make Style Transfer']
]

PARAMS_KB = [['max_length', 'no_repeat_ngram_size'],
             ['do_sample', 'top_k', 'top_p'],
             ['temperature', 'num_return_sequences'],
             ['device', 'is_always_use_length'],
             ['length_generate', 'Back']]

STYLE_KB = [['upload style image', 'upload content image'],
            ['style params', 'make transfer'], ['Back']]

BACK_KB = [['Back', 'Main menu'],]


PARAMS_STYLE_MAIN_KB = [["Layers Vgg", "Model Params"],
                        ["Back", "Main menu"]]

PARAMS_ST_CT_LAYERS_KB = [["Style layers", "Content layers"],
                          ["Back", "Reset layers", "Main menu"]]

PARAMS_LAYERS_ALL_KB = [["conv1_1(3, 64)", "conv1_2(64, 64)", "conv2_1(64,128)"],
                       ["conv2_2(128,128)", "conv3_1(128,256)", "conv3_2(256,256)"],
                       ["conv3_3(256,256)", "conv3_4(256,256)", "conv4_1(256,512)"],
                       ["conv4_2(512,512)", "conv4_3(512,512)", "conv4_4(512,512)"],
                       ["conv5_1(512,512)", "conv5_2(512,512)"],
                       ["conv5_3(512,512)", "conv5_4(512,512)"],
                       ["Back"]]


# PARAMS_VGG_LAYERS_KB = [["conv1_1", "conv2_1"],
#                           ["conv3_1", "conv4_1"],
#                           ["conv4_2", "conv5_1"], #"conv4_2" - content layer
#                           ["Back", "Main menu"]]




PARAMS_STYLE_KB = [["epoch numbers", "show cost [steps]"],
                   ["device[0:cpu, 1:cuda]", "image size"],
                   ["Back", "Main menu"]]


kb_remove = ReplyKeyboardRemove()


class KeyBoardGPT:
    def start_kb(self):
        markup = ReplyKeyboardMarkup(START_KB, one_time_keyboard=True)
        return markup


    def params_kb(self):
        markup = ReplyKeyboardMarkup(PARAMS_KB, one_time_keyboard=True)
        return markup


    def style_kb(self):
        markup = ReplyKeyboardMarkup(STYLE_KB, one_time_keyboard=True)
        return markup


    def back_kb(self):
        markup = ReplyKeyboardMarkup(BACK_KB, one_time_keyboard=True)
        return markup


    def params_style_main_kb(self):
        markup = ReplyKeyboardMarkup(PARAMS_STYLE_MAIN_KB, one_time_keyboard=True)
        return markup


    def params_st_ct_layers_kb(self):
        markup = ReplyKeyboardMarkup(PARAMS_ST_CT_LAYERS_KB, one_time_keyboard=True)
        return markup

    def params_layers_all_kb(self):
        markup = ReplyKeyboardMarkup(PARAMS_LAYERS_ALL_KB, one_time_keyboard=True)
        return markup


    def params_style_kb(self):
        markup = ReplyKeyboardMarkup(PARAMS_STYLE_KB, one_time_keyboard=True)
        return markup

