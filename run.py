#!/home/TeleBotGPT_VGG/venv/bin/python

from config import TOKEN, IP_ADDRESS, PORT

from db.db_connect import DBConnect
from db.db_gpt_manager import DBGpt
from db.db_vgg_manager import DBVgg

from lib.model_gpt import model
from lib.model_t_style import VggModel, ImagePreprocessing, layers_dict

import logging
import threading

from telegram.ext import (Updater,
                          CommandHandler,
                          MessageHandler,
                          Filters,
                          ConversationHandler)

from keyboard import (KeyBoard,
                      kb_remove,
                      KB_RE_GPT,
                      KB_RE_VGG_BASE,
                      GPT_RE_PARAM,
                      KB_RE_VGG_LAYERS)

GPT_MAIN_MENU, GPT_TALK_MENU, GPT_PARAM_MENU = range(3)
VGG_MAIN_MENU, VGG_UPLOAD_STYLE, VGG_UPLOAD_CONTENT = range(3, 6)
VGG_PARAMS_MENU, VGG_LAYERS_MENU, VGG_STYLE_PARAM, VGG_CONTENT_PARAM = range(6, 10)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

db_connect = DBConnect()
db_gpt = DBGpt()
db_vgg = DBVgg()

kb = KeyBoard()


def help_command(bot, update):
    text = "Это пробная версия бота, работающая на модели GPT2 от Сбербанка.\n" \
           "Дотюненая на отдельных датасетах by Yuriy Grossmend.\n" \
           "Так же бот имеет на борту модель VGG19, выполняющую задачу Style Transfer"
    bot.message.reply_text(text)


def help(bot, update):
    markup = kb.start_kb()

    bot.message.reply_text("<b>Модель GPT3:</b>\n"
                           "<b>Параметры:</b>\n"
                           "<b>max_length</b> - Максимальное количество токенов,\n"
                           "которые запоминает модель.\n"
                           "Рекомендуемый параметр - 256. "
                           "Максимум - 512. Будет больше - будет 512.\n"
                           "<b>no_repeat_ngram_size</b> - Штрафует за количество повторов.\n"
                           "<b>do_sample</b> - Значение может быть 0 или 1.\n"
                           "Если мы включаем - ставим 1, то мы автоматически отключаем параметры <b>top_p</b> и "
                           "<b>top_k</b>.\n"
                           "Если у нас 1 - мы начинаем выбирать из генерируемых слов, следующее слово на основе "
                           "условной вероятности.\n"
                           "Усиливает вероятность слов, которые нам больше подходят.\n"
                           "<b>top_k</b> - Количество слов из которых потом будет выбираться то - которое нам подходит.\n"
                           "<b>top_p</b> - То же самое, что и <b>top_k</b> - только в % (от 0 до 1)\n"
                           "Чем больше % - тем меньшее количество слов будет участвовать в дальнейшей выборке для "
                           "поиска нужного слова.\n"
                           "<b>temperature</b> - Значение от 0 до 3(Если больше - будет 3).\n"
                           "Как говорит <b>Михаил Константинов</b> - \"Температура - это то, как галлюцинирует ваша модель.\"\n"
                           "0 - Модель полностью адекватна и скучна.\n"
                           "3 - бред - бредом.\n"
                           "<b>num_return_sequences</b> - Количество сгенерированных возвращаемых предложений."
                           "Среди которых потом модеь  выберет подходящее предложение.\n"
                           "<b>device</b> - 0 - модель работает на CPU. 1 - модель работает на GPU.\n"
                           "У нас есть только CPU. Тут вариантов не много(\n"
                           "<b>is_always_use_length</b> - Значение может быть 0 или 1. Если выбираем 0 - генерируется\n"
                           "ответ любой длины.\n"
                           "<b>length_generate</b> - Длина генерируемых ботом предложений.\n"
                           "1 - Короткий ответ.\n"
                           "2 - Средний ответ.\n"
                           "3 - Длинный ответ.",  parse_mode='HTML')


    bot.message.reply_text("<b>Модель Vgg19</b>\n"
                           "Для работы с моделью сначала нужно загрузить два изображения.\n"
                           "То - которое будет отвечать за формирование стиля.\n"
                           "И то - которое будет отвечать за контент.\n"
                           "<b>Upload Style Image</b> - грузим наш стиль.\n"
                           "<b>Upload Content Image</b> - грузим наш контент.\n"
                           "\n<b>Параметры:</b>\n"
                           "Параметры разделены на 2 части.\n"
                           "<b>1 Часть - глобальные параметры.</b>\n"
                           "Те - которые используются непосредственно при генерации нового изображения.\n"
                           "<b>epoch number</b> - количество эпох.\n"
                           "Для генерации нормального результата нужно, как правило,"
                           "около 500-1000 эпох.\n"
                           "Но на CPU мы будем ждать вечность этого результата.\n"
                           "Поэтому рекомендуемый параметр - 20-50 эпох.\n"
                           "Ничего нормального из этого не получится.\n"
                           "Но как по другому без CUDA - ?\n"
                           "<b>show cost</b> - раз во сколько эпох мы будем смотреть наши лоссы.\n"
                           "Но на боте это опять же не работает.\n"
                           "Потому что пока полностью не пройдем процесс - результат не вернется.\n"
                           "<b>device</b> - как было написано выше - у нас только CPU.\n"
                           "<b>image size</b> - размер генерируемого изображения. "
                           "Максимальное значение - 512.\n"
                           "<b>2 Часть - параметры слоев.</b>\n"
                           "На данный момент стоят принятые общие параметры.\n"
                           "Но с ними можно играться и регулировать их.\n"
                           "Правда как посмотреть результат, если нормального изображения"
                           "не сгенерируешь ...вопрос....", reply_markup = markup, parse_mode='HTML')


def start(bot, update):
    markup = kb.start_kb()
    if db_connect.check_connect(bot.message.chat.id):

        bot.message.reply_text(f"Привет, {bot.message.from_user.first_name}.\n"
                               f"На данный момент можно поговорить с чат-ботом GPT3\n"
                               f"или сделать Style Transfer.\n"
                               f"Выбери один из пунктов меню.", reply_markup=markup, )
    else:
        bot.message.reply_text(f"Привет, {bot.message.from_user.first_name}.\n"
                               f"произошла ошибка при записи/считывании Вас из базы,\n"
                               f"обратитесь к разработчику.", reply_markup=markup, )


# -------------gpt main menu---------------------------------
def gpt_main_menu(bot, update):
    if db_gpt.check_gpt_params(bot.message.chat.id):
        markup = kb.gpt_main_kb()
        bot.message.reply_text(f"Вы находитесь в GPT3 меню.\n"
                               f"", reply_markup=markup)
    else:
        markup = kb.start_kb()
        bot.message.reply_text(f"Ошибка доступа к GPT3 боту.", reply_markup=markup)

    return GPT_MAIN_MENU


# -------------end main menu----------------------------------


# -------------gpt params block-------------------------------
def gpt_params(bot, update):
    update.user_data.clear()
    params = db_gpt.get_params(bot.message.chat.id)
    if params:
        markup = kb.gpt_params_kb()
        bot.message.reply_text(f'На данный момент GPT3 имеет такие параметры: \n'
                               f'max_length: {params["max_length"]}\n'
                               f'no_repeat_ngram_size: {params["no_repeat_ngram_size"]}\n'
                               f'do_sample: {str(params["do_sample"])}\n'
                               f'top_k: {params["top_k"]}\n'
                               f'top_p: {params["top_p"]}\n'
                               f'temperature: {params["temperature"]}\n'
                               f'num_return_sequences: {params["num_return_sequences"]}\n'
                               f'device: {params["device"]}\n'
                               f'is_always_use_length: {params["is_always_use_length"]}\n'
                               f'length_generate: {params["length_generate"]}', reply_markup=markup)
    else:
        markup = kb.gpt_main_kb()
        bot.message.reply_text(f"Ошибка доступа к параметрам GPT3.\n"
                               f"Возврат в предыдущее меню.", reply_markup=markup)
        return GPT_MAIN_MENU
    return GPT_PARAM_MENU


def gpt_old_param(bot, update):
    user_data = update.user_data
    text = bot.message.text

    print(user_data)

    user_data['choice'] = text
    category = user_data['choice']

    print(category)

    user_data[category] = text
    del user_data['choice']

    params = db_gpt.get_params(bot.message.chat.id)
    if params:
        bot.message.reply_text(f"Старое значение: {str(params[text])}")
        markup = kb.gpt_params_kb()
        bot.message.reply_text("Введите новое значение: ", reply_markup=markup)
    else:
        markup = kb.gpt_main_kb()
        bot.message.reply_text("Неверный запрос.", reply_markup=markup)
        return GPT_MAIN_MENU
    return GPT_PARAM_MENU


def gpt_new_param(bot, update):
    try:
        print("update.user_data:", update.user_data)
        key = [val for val in update.user_data.keys()][-1]
        print(key)
        text = bot.message.text
        update.user_data[key] = text
        up_param = [key, update.user_data[key], bot.message.chat.id]
        print(up_param)

        db_gpt.save_params(up_param)
        markup = kb.gpt_params_kb()
        bot.message.reply_text(f"Новое значение: {update.user_data[key]}")

        if update.user_data:
            del update.user_data[key]
        bot.message.reply_text(f"Можно еще что-то поменять или жмякнуть Back", reply_markup=markup)
    except Exception as ex:
        markup = kb.gpt_params_kb()
        bot.message.reply_text(f"{ex}: Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз, либо нажмите Back", reply_markup=markup)
    return GPT_PARAM_MENU


# -------------end gpt params block-------------------------------


# -------------------gpt reset all params to factory default------
def gpt_reset_params(bot, update):
    try:
        if db_gpt.reset_params(bot.message.chat.id):
            bot.message.reply_text("Все параметры сброшены до \"Заводских\" настроек.")

            markup = kb.gpt_params_kb()
            params = db_gpt.get_params(bot.message.chat.id)
            if params:
                bot.message.reply_text(f'На данный момент GPT3 имеет такие параметры: \n'
                                       f'max_length: {params["max_length"]}\n'
                                       f'no_repeat_ngram_size: {params["no_repeat_ngram_size"]}\n'
                                       f'do_sample: {str(params["do_sample"])}\n'
                                       f'top_k: {params["top_k"]}\n'
                                       f'top_p: {params["top_p"]}\n'
                                       f'temperature: {params["temperature"]}\n'
                                       f'num_return_sequences: {params["num_return_sequences"]}\n'
                                       f'device: {params["device"]}\n'
                                       f'is_always_use_length: {params["is_always_use_length"]}\n'
                                       f'length_generate: {params["length_generate"]}', reply_markup=markup)
    except Exception as ex:
        bot.message.reply_text(f"{ex}: Ошибка сброса параметров.")

    return GPT_PARAM_MENU


# -------------------end gpt reset all params to factory default------


# -------------------gpt talk block-----------------------------------
def start_talk(bot, update):
    markup = kb.gpt_talk_kb()
    bot.message.reply_text('Уже можно общаться ;)', reply_markup=markup)

    return GPT_TALK_MENU


def gpt_talk(bot, update):
    params = db_gpt.get_params(bot.message.chat.id)
    markup = kb.gpt_talk_kb()
    if params:
        answer = model.get_response(bot.message.chat.id, bot.message.text, params, db_gpt)
        bot.message.reply_text(answer, reply_markup=markup)
    else:
        bot.message.reply_text("Что то в общении пошло не так...", reply_markup=markup)

    return GPT_TALK_MENU


# -------------------end gpt2 talk block---------------------------


# -----------------gpt clear all dialogues--------------------------
def gpt_clear_dialogues(bot, update):
    markup = kb.gpt_main_kb()
    if db_gpt.clear_dialogue(bot.message.chat.id):
        bot.message.reply_text('Все диалоги удалены из базы.', reply_markup=markup)
    else:
        bot.message.reply_text('Что-то пошло не так...', reply_markup=markup)
    return GPT_TALK_MENU


# -----------------end gpt clear all dialogues----------------------


# -------------------vgg main menu-----------------------------
def vgg_main_menu(bot, update):
    markup = kb.vgg_main_kb()

    bot.message.reply_text(f"Вы находитесь в VGG19 меню.\n"
                           f"", reply_markup=markup)

    if db_vgg.add_base_params(bot.message.chat.id):
        print("Hello")
        base_style = VggModel.base_style_layer.copy()
        base_content = VggModel.base_cont_layer.copy()

        # add to VGG base values for content&style layers
        if not db_vgg.add_base_layers_params(bot.message.chat.id, base_style, base_content):
            bot.message.reply_text(f"Ошибка при добавлении в layers Vgg db!", reply_markup=markup, )

    else:
        markup = kb.vgg_main_kb()
        bot.message.reply_text(f"Ошибка доступа к VGG19.", reply_markup=markup)

    return VGG_MAIN_MENU


# ------------------end vgg main menu--------------------------


# ----------------vgg upload style image ----------------------
def style_image(bot, update):
    markup = kb.back_kb()
    bot.message.reply_text("Загрузите изображение для формирования стиля.", reply_markup=markup)

    return VGG_UPLOAD_STYLE


def upload_style_image(bot, update):
    markup = kb.back_kb()

    file_style = update.bot.get_file(bot.message.photo[-1].file_id)

    if db_vgg.save_file_link(bot.message.chat.id, file_path=file_style['file_path'], is_style=True):
        bot.message.reply_text("Файл для формирования стиля изображения получен.\n"
                               "Для возврата в предыдущее меню, нажмите 'Back'",
                               reply_markup=markup)
    else:
        bot.message.reply_text("Что то пошло не так...\n"
                               "Попробуйте загрузить изображение снова или нажмите 'Back'", reply_markup=markup)

    return VGG_UPLOAD_STYLE


# -------------end vgg upload style image --------------------


# ---------------vgg upload content image ----------------------
def content_image(bot, update):
    markup = kb.back_kb()
    bot.message.reply_text("Загрузите изображение для формирования Content - a.", reply_markup=markup)

    return VGG_UPLOAD_CONTENT


def upload_content_image(bot, update):
    markup = kb.back_kb()

    file_content = update.bot.get_file(bot.message.photo[-1].file_id)

    if db_vgg.save_file_link(bot.message.chat.id, file_path=file_content['file_path'], is_style=False):
        bot.message.reply_text("Файл для формирования контента изображения получен.\n"
                               "Для возврата в предыдущее меню, нажмите 'Back'",
                               reply_markup=markup)
    else:
        bot.message.reply_text("Что то пошло не так...\n"
                               "Попробуйте загрузить изображение снова или нажмите 'Back'", reply_markup=markup)

    return VGG_UPLOAD_CONTENT


# -----------end vgg upload content image ----------------------


# --------------vgg params main menu ----------------------
def vgg_params_main(bot, update):
    markup = kb.vgg_params_main_kb()
    bot.message.reply_text("Вы находитесь в основном меню параметров VGG19", reply_markup=markup)

    return VGG_PARAMS_MENU


# ---------end vgg params main menu -----------------------


# ---------vgg base transfer params -----------------------
def vgg_transfer_params(bot, update):
    update.user_data.clear()
    markup = kb.vgg_params_transfer_kb()
    base_params = db_vgg.get_base_vgg_params(bot.message.chat.id)

    if base_params:
        bot.message.reply_text(f'На данный момент VGG19 имеет такие базовые параметры: \n'
                               f'epoch number (steps): {base_params["epoch number"]}\n'
                               f'show cost every: {base_params["show cost [steps]"]} steps\n'
                               f'device: {str(base_params["device[0:cpu, 1:cuda]"])} [0: cpu, 1: cuda] (Only CPU now)\n'
                               f'image_size: {base_params["image size"]}\n'
                               f'Что будем менять?', reply_markup=markup)
    else:
        bot.message.reply_text('Ошибка считывания базовых параметров,\n'
                               'используемых для трансформации изображения.', reply_markup=markup)

    return VGG_PARAMS_MENU


# ------end vgg base transfer params -----------------------


# ---------- vgg old base params -----------------------
def vgg_old_base_params(bot, update):
    markup = kb.vgg_params_transfer_kb()
    user_data = update.user_data
    text = bot.message.text

    update.user_data['choice'] = text
    category = user_data['choice']

    user_data[category] = text
    del user_data['choice']

    base_params = db_vgg.get_base_vgg_params(bot.message.chat.id)
    if base_params:
        bot.message.reply_text(f"Старое значение: {str(base_params[text])}")
        bot.message.reply_text("Введите новое значение: ", reply_markup=markup)
    else:
        bot.message.reply_text('Ошибка считывания старого параметра.', reply_markup=markup)

    return VGG_PARAMS_MENU


# ----------end vgg old base params -----------------------

# ---------- vgg NEW base params -----------------------
def vgg_new_base_params(bot, update):
    markup = kb.vgg_params_transfer_kb()

    try:
        key = [val for val in update.user_data.keys()][0]
        text = bot.message.text
        update.user_data[key] = text
        up_param = [key, update.user_data[key], bot.message.chat.id]

        if up_param[0] == 'device[0:cpu, 1:cuda]':
            bot.message.reply_text(f"Не, сорри, у нас только cpu. Тут как-бы без вариантов", reply_markup=markup)
            if update.user_data:
                del update.user_data[key]
                return VGG_PARAMS_MENU

        db_vgg.save_vgg_base_params(up_param)
        bot.message.reply_text(f"Новое значение: {update.user_data[key]}", reply_markup=markup)

        if update.user_data:
            del update.user_data[key]

        return VGG_PARAMS_MENU
    except Exception as ex:

        bot.message.reply_text(f"{ex}: Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз, либо нажмите Back", reply_markup=markup)
        return VGG_PARAMS_MENU


# ----------end vgg NEW base params -----------------------


# ---------- vgg layer params -----------------------
def vgg_layers(bot, update):
    markup = kb.vgg_layers_kb()
    update.user_data.clear()

    bot.message.reply_text(f'Тут можно назначить Conv2d слои для CONTENT и STYLE изображения.\n'
                           f'0 - Это значит, что слой вообще не участвует в Style Transfer\n'
                           f'1(или любая другая цифра) - Слой активен.', reply_markup=markup)

    return VGG_LAYERS_MENU


# --------end vgg layer params -----------------------


# ----------- vgg old style params -------------------
def style_layers(bot, update):
    markup = kb.vgg_layers_params_kb()

    bot.message.reply_text(f'Тут можно назначить Conv2d слои для STYLE изображения.\n'
                           f'0 - Это значит, что слой вообще не участвует в Style Transfer\n'
                           f'1(или любая другая цифра) - Слой активен.')

    sl = db_vgg.get_style_params(bot.message.chat.id, style_layer=True)
    if sl:
        bot.message.reply_text(f'Текущие значения для STYLE Image следующие:\n'
                               f'conv1_1(3, 64):   {sl[0]}\n'
                               f'conv1_2(64, 64):  {sl[1]}\n'
                               f'conv2_1(64, 128): {sl[2]}\n'
                               f'conv2_2(128,128): {sl[3]}\n'
                               f'conv3_1(128,256): {sl[4]}\n'
                               f'conv3_2(256,256): {sl[5]}\n'
                               f'conv3_3(256,256): {sl[6]}\n'
                               f'conv3_4(256,256): {sl[7]}\n'
                               f'conv4_1(256,512): {sl[8]}\n'
                               f'conv4_2(512,512): {sl[9]}\n'
                               f'conv4_3(512,512): {sl[10]}\n'
                               f'conv4_4(512,512): {sl[11]}\n'
                               f'conv5_1(512,512): {sl[12]}\n'
                               f'conv5_2(512,512): {sl[13]}\n'
                               f'conv5_3(512,512): {sl[14]}\n'
                               f'conv5_4(512,512): {sl[15]}', reply_markup=markup)
    else:
        bot.message.reply_text("Ошибка доступа к параметрам слоёв для STYLE IMAGE.\n"
                               "Попробуйте снова или нажмите 'Back'", reply_markup=markup)
    return VGG_STYLE_PARAM


def vgg_old_style_param(bot, update):
    markup = kb.vgg_layers_params_kb()

    user_data = update.user_data
    text = bot.message.text

    update.user_data['choice'] = text
    category = user_data['choice']

    user_data[category] = text
    del user_data['choice']

    sl = db_vgg.get_style_params(bot.message.chat.id, style_layer=True)
    if sl:
        params = {}
        for idx, val in enumerate(sl):
            params[layers_dict[str(idx)]] = val

        bot.message.reply_text(f"Старое значение: {str(params[text])}")
        bot.message.reply_text("Введите новое значение: ", reply_markup=markup)
    else:
        bot.message.reply_text("Неверный запрос.", reply_markup=markup)
    return VGG_STYLE_PARAM


# --------end vgg old style params -------------------


# ----------- vgg new style params -------------------
def vgg_new_style_param(bot, update):
    markup = kb.vgg_layers_params_kb()
    key = [val for val in update.user_data.keys()][0]
    text = bot.message.text
    update.user_data[key] = text
    up_param = [key, update.user_data[key], bot.message.chat.id]

    if db_vgg.save_layer_param(up_param, style_layer=True):
        bot.message.reply_text(f"Новое значение: {float(bool(int(update.user_data[key])))}", reply_markup=markup)
        if update.user_data:
            del update.user_data[key]
    else:
        bot.message.reply_text(f"Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз либо нажмите Back", reply_markup=markup)
    return VGG_STYLE_PARAM


# --------end vgg new style params -------------------


# ----------- vgg old content params -------------------
def content_layers(bot, update):
    markup = kb.vgg_layers_params_kb()

    bot.message.reply_text(f'Тут можно назначить Conv2d слои для CONTENT изображения.\n'
                           f'0 - Это значит, что слой вообще не участвует в Style Transfer\n'
                           f'1(или любая другая цифра) - Слой активен.')
    cl = db_vgg.get_style_params(bot.message.chat.id, style_layer=False)
    if cl:
        bot.message.reply_text(f'Текущие значения для CONTENT Image следующие:\n'
                               f'conv1_1(3, 64):   {cl[0]}\n'
                               f'conv1_2(64, 64):  {cl[1]}\n'
                               f'conv2_1(64, 128): {cl[2]}\n'
                               f'conv2_2(128,128): {cl[3]}\n'
                               f'conv3_1(128,256): {cl[4]}\n'
                               f'conv3_2(256,256): {cl[5]}\n'
                               f'conv3_3(256,256): {cl[6]}\n'
                               f'conv3_4(256,256): {cl[7]}\n'
                               f'conv4_1(256,512): {cl[8]}\n'
                               f'conv4_2(512,512): {cl[9]}\n'
                               f'conv4_3(512,512): {cl[10]}\n'
                               f'conv4_4(512,512): {cl[11]}\n'
                               f'conv5_1(512,512): {cl[12]}\n'
                               f'conv5_2(512,512): {cl[13]}\n'
                               f'conv5_3(512,512): {cl[14]}\n'
                               f'conv5_4(512,512): {cl[15]}', reply_markup=markup)
    else:
        bot.message.reply_text("Ошибка доступа к параметрам слоёв для CONTENT IMAGE.\n"
                               "Попробуйте снова или нажмите 'Back'", reply_markup=markup)
    return VGG_CONTENT_PARAM


def vgg_old_content_param(bot, update):
    markup = kb.vgg_layers_params_kb()

    user_data = update.user_data
    text = bot.message.text
    update.user_data['choice'] = text
    category = user_data['choice']

    user_data[category] = text
    del user_data['choice']

    cl = db_vgg.get_style_params(bot.message.chat.id, style_layer=False)
    if cl:
        params = {}
        for idx, val in enumerate(cl):
            params[layers_dict[str(idx)]] = val

        bot.message.reply_text(f"Старое значение: {str(params[text])}")
        bot.message.reply_text("Введите новое значение: ", reply_markup=markup)
    else:
        bot.message.reply_text("Неверный запрос.", reply_markup=markup)
    return VGG_CONTENT_PARAM


# --------end vgg old content params -------------------


# ----------- vgg new content params -------------------
def vgg_new_content_param(bot, update):
    markup = kb.vgg_layers_params_kb()

    key = [val for val in update.user_data.keys()][0]
    text = bot.message.text
    update.user_data[key] = text
    up_param = [key, update.user_data[key], bot.message.chat.id]

    if db_vgg.save_layer_param(up_param, style_layer=False):
        bot.message.reply_text(f"Новое значение: {float(bool(int(update.user_data[key])))}", reply_markup=markup)
        if update.user_data:
            del update.user_data[key]
    else:
        bot.message.reply_text(f"Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз, либо нажмите Back", reply_markup=markup)

    return VGG_CONTENT_PARAM


# --------end vgg new content params -------------------


# -----------reset all layers params --------------------
def reset_layers(bot, update):
    base_style = VggModel.base_style_layer.copy()
    base_cont = VggModel.base_cont_layer.copy()
    markup = kb.vgg_layers_kb()
    if db_vgg.reset_layers(bot.message.chat.id, base_style, base_cont):
        bot.message.reply_text("Слои VGG в исходном состоянии.", reply_markup=markup)
    else:
        bot.message.reply_text("Что-то пошло не так.", reply_markup=markup)
    return VGG_LAYERS_MENU


# --------end reset all layers params --------------------


# -----------------run style transfer------------------------
# @run_async
def run_transfer(bot, update):
    def closure():
        base_params = db_vgg.get_base_vgg_params(bot.message.chat.id)
        stile_link = db_vgg.load_file_link(bot.message.chat.id, is_style=True)
        content_link = db_vgg.load_file_link(bot.message.chat.id)

        markup = kb.vgg_main_kb()

        if stile_link is None or content_link is None:
            bot.message.reply_text("Ошибка загрузки одного из изображений.\n"
                                   "", reply_markup=markup)
            return VGG_MAIN_MENU

        image_preprocess = ImagePreprocessing(base_params["image size"])
        style = image_preprocess.image_loader(file_path=stile_link)
        content = image_preprocess.image_loader(file_path=content_link)
        input_img = content

        if style is None or content is None:
            bot.message.reply_text("Ошибка препроцессинга изображения\n"
                                   "Загрузите другие изображения или перегрузите текущие.", reply_markup=markup)
            return VGG_MAIN_MENU

        vgg_model = VggModel()
        stl_layers = db_vgg.get_style_params(bot.message.chat.id, style_layer=True)
        con_layers = db_vgg.get_style_params(bot.message.chat.id, style_layer=False)

        bot.message.reply_text("Начинаем style-transfer.\n"
                               "Это займет какое то время....\n"
                               "Можно пока с ботом поговорить :)")

        output = vgg_model.run_style_transfer(content, style, input_img, num_steps=base_params["epoch number"],
                                              show_every=base_params["show cost [steps]"], style_layers=stl_layers,
                                              content_layers=con_layers)
        try:
            img = image_preprocess.im_detorch(output)
            img_byte_arr = image_preprocess.im_to_bytearray(img)
            update.bot.send_photo(chat_id=bot.message.chat.id, photo=img_byte_arr)
            bot.message.reply_text("Готово!", reply_markup=markup)
        except Exception as ex:
            bot.message.reply_text(f"{ex}: Ошибка при работе с моделью STYLE-TRANSFER.", reply_markup=markup)
        return VGG_MAIN_MENU

    thr1 = threading.Thread(target=closure)
    thr1.start()
    return VGG_MAIN_MENU


def exit_ai(bot, update):
    user_data = update.user_data
    if 'choice' in user_data:
        del user_data['choice']

    bot.message.reply_text(f"До новых встреч )", reply_markup=kb_remove)
    user_data.clear()
    return gpt_main_menu


# --------------end run style transfer------------------------


def main():
    updater = Updater(TOKEN)




    gpt_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.regex('^GPT3 Menu$'), gpt_main_menu)],
        states={
            GPT_MAIN_MENU: [
                MessageHandler(Filters.regex('^GPT3 Menu$'), gpt_main_menu),
                MessageHandler(Filters.regex('^Start Talk$'), start_talk),  # temp!!!!!!!!!
                MessageHandler(Filters.regex('^GPT3 Params$'), gpt_params),
                MessageHandler(Filters.regex('^Clear Dialoque$'), gpt_clear_dialogues),
                MessageHandler(Filters.regex('^Start Menu$'), start)
            ],
            GPT_TALK_MENU: [
                MessageHandler(Filters.regex('^Start Menu$'), start),
                MessageHandler(Filters.regex('^Start Talk$'), start_talk),
                MessageHandler(Filters.regex('^GPT3 Menu$'), gpt_main_menu),
                MessageHandler(Filters.regex('^Clear Dialoque$'), gpt_clear_dialogues),
                MessageHandler(Filters.regex('^GPT3 Params$'), gpt_params),
                MessageHandler(Filters.text & ~(Filters.command | Filters.regex(KB_RE_GPT)), gpt_talk),
            ],
            GPT_PARAM_MENU: [
                MessageHandler(Filters.regex('^Back$'), gpt_main_menu),
                MessageHandler(Filters.regex(GPT_RE_PARAM), gpt_old_param),
                MessageHandler(Filters.regex('^reset params to factory default$'), gpt_reset_params),
                MessageHandler(Filters.text & ~(Filters.command | Filters.regex(GPT_RE_PARAM)), gpt_new_param)

            ]

        },
        fallbacks=[]
    )

    vgg_handler = ConversationHandler(
        entry_points=[MessageHandler(Filters.regex('^Style Transfer Menu$'), vgg_main_menu)],
        states={
            VGG_MAIN_MENU: [
                MessageHandler(Filters.regex('^Start Menu$'), start),
                MessageHandler(Filters.regex('^Style Transfer Menu$'), vgg_main_menu),

                MessageHandler(Filters.regex('^Back$'), vgg_main_menu),
                MessageHandler(Filters.regex('^Upload Style Image$'), style_image),
                MessageHandler(Filters.regex('^Upload Content Image$'), content_image),
                MessageHandler(Filters.regex('^VGG19 params$'), vgg_params_main),
                MessageHandler(Filters.regex('^Make Style Transfer$'), run_transfer, run_async=True)
            ],
            VGG_UPLOAD_STYLE: [
                MessageHandler(Filters.photo, upload_style_image),
                MessageHandler(Filters.regex('^Back$'), vgg_main_menu)
            ],
            VGG_UPLOAD_CONTENT: [
                MessageHandler(Filters.regex('^Back$'), vgg_main_menu),
                MessageHandler(Filters.photo, upload_content_image),
            ],
            VGG_PARAMS_MENU: [
                MessageHandler(Filters.regex('^Back$'), vgg_params_main),
                MessageHandler(Filters.regex('^Main VGG19 Menu$'), vgg_main_menu),
                MessageHandler(Filters.regex('^VGG19 Layers Params$'), vgg_layers),
                MessageHandler(Filters.regex('^Base Transfer Params$'), vgg_transfer_params),

                MessageHandler(Filters.regex(KB_RE_VGG_BASE), vgg_old_base_params),
                MessageHandler(Filters.text & ~(Filters.command | Filters.regex(KB_RE_VGG_BASE)), vgg_new_base_params),
            ],
            VGG_LAYERS_MENU: [
                MessageHandler(Filters.regex('^Back$'), vgg_params_main),
                MessageHandler(Filters.regex('^Main VGG19 Menu$'), vgg_main_menu),
                MessageHandler(Filters.regex('^Reset layers$'), reset_layers),
                MessageHandler(Filters.regex('^Content layers$'), content_layers),
                MessageHandler(Filters.regex('^Style layers$'), style_layers),

            ],
            VGG_STYLE_PARAM: [
                MessageHandler(Filters.regex('^Back$'), vgg_layers),
                MessageHandler(Filters.regex(KB_RE_VGG_LAYERS), vgg_old_style_param),
                MessageHandler(Filters.text & ~(Filters.command | Filters.regex(KB_RE_VGG_LAYERS)),
                               vgg_new_style_param),
            ],
            VGG_CONTENT_PARAM: [
                MessageHandler(Filters.regex('^Back$'), vgg_layers),
                MessageHandler(Filters.regex(KB_RE_VGG_LAYERS), vgg_old_content_param),
                MessageHandler(Filters.text & ~(Filters.command | Filters.regex(KB_RE_VGG_LAYERS)),
                               vgg_new_content_param),
            ],

        },
        fallbacks=[]
    )

    dispatcher = updater.dispatcher
    dispatcher.add_handler(gpt_handler)
    dispatcher.add_handler(vgg_handler)
    dispatcher.add_handler(CommandHandler('start', start))
    dispatcher.add_handler(CommandHandler('help', help))
    dispatcher.add_handler(MessageHandler(Filters.regex('^Exit$'), exit_ai))

    updater.start_polling()

    # updater.start_webhook(
    #     listen='0.0.0.0',
    #     port=PORT,
    #     url_path=TOKEN,
    #     webhook_url= f'https://{IP_ADDRESS}/{TOKEN}',
    #     cert='cert.pem')

    # updater.start_webhook(
    #     listen='0.0.0.0',
    #     port=PORT,
    #     url_path=TOKEN,
    #     key='private.key',
    #     cert='cert.pem',
    #     webhook_url=f'https://{IP_ADDRESS}:8443/{TOKEN}')

    updater.idle()


if __name__ == '__main__':
    main()
