from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove

KB_RE_GPT = '^(GPT3 Menu|Style Transfer Menu|Exit|' \
            'Start Talk|GPT3 Params|Clear Dialoque|Start Menu)$'

GPT_RE_PARAM = '^(max_length|no_repeat_ngram_size|do_sample|top_k|num_return_sequences|' \
               'top_p|temperature|device|is_always_use_length|' \
               'length_generate)$'

KB_RE_VGG_BASE = '^(epoch number|show cost \[steps\]|image size|device\[0:cpu, 1:cuda\])$'

KB_RE_VGG_LAYERS = '^(conv1_1\(3, 64\)|conv1_2\(64, 64\)|conv2_1\(64,128\)|conv2_2\(128,128\)' \
                   '|conv3_1\(128,256\)|conv3_2\(256,256\)|conv3_3\(256,256\)|conv3_4\(256,256\)' \
                   '|conv4_1\(256,512\)|conv4_2\(512,512\)|conv4_3\(512,512\)|conv4_4\(512,512\)' \
                   '|conv5_1\(512,512\)|conv5_2\(512,512\)|conv5_3\(512,512\)|conv5_4\(512,512\))$'


START_BUTTON_KB = [['/start']]


# ------------- bot start menu-----------
START_MENU_KB = [['GPT3 Menu'],
                 ['Style Transfer Menu'],
                 ['Exit']]

# ------------------gpt model----------------------
GPT_MAIN_KB = [['Start Talk', 'GPT3 Params'],
               ['Clear Dialoque', 'Start Menu']]

GPT_TALK_KB = [['Clear Dialoque', 'GPT3 Params'],
               ['Start Menu', 'GPT3 Menu']]

START_KB = [['Just Talk', 'Params'],
            ['Clear Talk Data', 'Exit'],
            ['Make Style Transfer']]

PARAMS_KB = [['reset params to factory default'],
             ['max_length', 'no_repeat_ngram_size'],
             ['do_sample', 'top_k', 'top_p'],
             ['temperature', 'num_return_sequences'],
             ['device', 'is_always_use_length'],
             ['length_generate', 'Back']]

# ---------------------vgg model--------------------------
VGG_MAIN_KB = [['Upload Style Image', 'Upload Content Image'],
               ['VGG19 params', 'Make Style Transfer'],
               ['Start Menu']]

BACK_KB = [['Back']]

VGG_MAIN_PARAMS_MENU_KB = [["VGG19 Layers Params", "Base Transfer Params"],
                           ["Main VGG19 Menu"]]

VGG_BASE_TRANSFER_PARAMS_KB = [["epoch number", "show cost [steps]"],
                               ["device[0:cpu, 1:cuda]", "image size"],
                               ["Back", "Main VGG19 Menu"]]

VGG_LAYERS_KB = [["Style layers", "Content layers"],
                 ["Back", "Reset layers", "Main VGG19 Menu"]]

VGG_LAYERS_PARAMS_KB = [["conv1_1(3, 64)", "conv1_2(64, 64)"],
                        ["conv2_1(64,128)", "conv2_2(128,128)"],
                        ["conv3_1(128,256)", "conv3_2(256,256)"],
                        ["conv3_3(256,256)", "conv3_4(256,256)"],
                        ["conv4_1(256,512)", "conv4_2(512,512)"],
                        ["conv4_3(512,512)", "conv4_4(512,512)"],
                        ["conv5_1(512,512)", "conv5_2(512,512)"],
                        ["conv5_3(512,512)", "conv5_4(512,512)"],
                        ["Back"]]

kb_remove = ReplyKeyboardRemove()


class KeyBoard:
    @staticmethod
    def start_button():
        markup = ReplyKeyboardMarkup(START_BUTTON_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def start_kb():
        markup = ReplyKeyboardMarkup(START_MENU_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def gpt_main_kb():
        markup = ReplyKeyboardMarkup(GPT_MAIN_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def gpt_talk_kb():
        markup = ReplyKeyboardMarkup(GPT_TALK_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def gpt_params_kb():
        markup = ReplyKeyboardMarkup(PARAMS_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def vgg_main_kb():
        markup = ReplyKeyboardMarkup(VGG_MAIN_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def back_kb():
        markup = ReplyKeyboardMarkup(BACK_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def vgg_params_main_kb():
        markup = ReplyKeyboardMarkup(VGG_MAIN_PARAMS_MENU_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def vgg_params_transfer_kb():
        markup = ReplyKeyboardMarkup(VGG_BASE_TRANSFER_PARAMS_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def vgg_layers_kb():
        markup = ReplyKeyboardMarkup(VGG_LAYERS_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup

    @staticmethod
    def vgg_layers_params_kb():
        markup = ReplyKeyboardMarkup(VGG_LAYERS_PARAMS_KB, one_time_keyboard=True, resize_keyboard=True)
        return markup