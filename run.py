#!/usr/bin/env python3

from config import TOKEN, IP_ADDRESS, PORT
from db.db_manager import DBManager, DBStyle_Transfer

from keyboard import KeyBoardGPT, kb_remove
from lib.model_gpt import model
# from lib.model_style import Vgg, ImagePreprocessing
from lib.model_style_tmp import VggModel, ImagePreprocessing, layers_dict


from io import BytesIO

import logging
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, ConversationHandler)

import io

CHOOSING, TYPING_REPLY, TYPING_CHOICE, CHANGING, NEW_VALUE = range(5)
STYLE_CONT_CHOOSE, UPLOAD_STYLE, UPLOAD_CONTENT, STYLE_PARAM, NEW_STYLE_VALUE = range(5, 10)
CONTENT_LAYERS_CHOICE, STYLE_LAYERS_CHOICE, RESET_LAYERS = range(10, 13)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

db_instance = DBManager()


def start(bot, update):
    kb = KeyBoardGPT()
    markup = kb.start_kb()
    bot.message.reply_text(f"Привет, {bot.message.from_user.first_name} {bot.message.from_user.last_name}. " +
                           f"Я бот GPT2.", reply_markup=markup,)

    args = [bot.message.chat.id, bot.message.from_user.first_name, bot.message.from_user.last_name]
    records = db_instance.check_connect(args[0], args)

    if len(records) > 0 and records[0][4] is not None:
        bot.message.reply_text(f"Мы уже с тобой беседовали. Длина сохраненного диалога: {len(records[0][4])}\n." +
                               f" Диалог можно либо продолжить, либо очистить")

    params = db_instance.get_params(bot.message.chat.id)
    bot.message.reply_text(f'На данный момент GPT2 имеет такие параметры: \n' +
            f'max_length: {params["max_length"]}, no_repeat_ngram_size: {params["no_repeat_ngram_size"]}\n' +
            f'do_sample: {str(params["do_sample"])}, top_k: {params["top_k"]}, top_p: {params["top_p"]}\n' +
            f'temperature: {params["temperature"]}, num_return_sequences: {params["num_return_sequences"]}\n' +
            f'device: {params["device"]}, is_always_use_length: {params["is_always_use_length"]}\n' +
            f'length_generate: {params["length_generate"]}')

    #add to VGG epoch, show_every and etc etc
    db_style = DBStyle_Transfer(bot.message.chat.id)
    if db_style.add_base_params() == False:
        bot.message.reply_text(f"Ошибка при добавлении в style-transfer db!", reply_markup=markup, )

    base_style = VggModel.base_style_layer.copy()
    base_content = VggModel.base_cont_layer.copy()

    #add to VGG base values for content&style layers
    if db_style.add_base_layers_params(base_style, base_content) == False:
        bot.message.reply_text(f"Ошибка при добавлении в layers Vgg db!", reply_markup=markup, )


    return CHOOSING


def help_command(bot, update):
    text = "Это пробная версия бота, работающая на модели GPT2 от сбербанка, дотюненая на отдельных датасетах by Yuriy Grossmend.\n" \
           "А так же бот имеет на борту модель VGG19, выполняющая задачу Style Transfer"
    bot.message.reply_text(text)


def main_menu(bot, update):
    kb = KeyBoardGPT()
    markup = kb.start_kb()
    bot.message.reply_text("Выберете пункт меню или просто начинайте разговаривать)", reply_markup=markup)

    return CHOOSING


#----------------gpt params change-------------------------
def gpt_params(bot, update):
    update.user_data.clear()
    params = db_instance.get_params(bot.message.chat.id)
    bot.message.reply_text(f'На данный момент GPT2 имеет такие параметры: \n' +
            f'max_length: {params["max_length"]}, no_repeat_ngram_size: {params["no_repeat_ngram_size"]}\n' +
            f'do_sample: {str(params["do_sample"])}, top_k: {params["top_k"]}, top_p: {params["top_p"]}\n' +
            f'temperature: {params["temperature"]}, num_return_sequences: {params["num_return_sequences"]}\n' +
            f'device: {params["device"]}, is_always_use_length: {params["is_always_use_length"]}\n' +
            f'length_generate: {params["length_generate"]}')

    kb = KeyBoardGPT()
    markup = kb.params_kb()
    bot.message.reply_text('Что будем менять?', reply_markup=markup)

    return TYPING_REPLY


def received_information(bot, update):
    user_data = update.user_data
    text = bot.message.text

    update.user_data['choice'] = text
    category = user_data['choice']

    user_data[category] = text
    del user_data['choice']

    params = db_instance.get_params(bot.message.chat.id)
    bot.message.reply_text(f"Старое значение: {str(params[text])}")

    kb = KeyBoardGPT()
    markup = kb.params_kb()
    bot.message.reply_text("Введите новое значение: ", reply_markup=markup )

    return NEW_VALUE


def changing_params(bot, update):
    kb = KeyBoardGPT()
    try:
        key = [val for val in update.user_data.keys()][0]
        text = bot.message.text
        update.user_data[key] = text

        up_param = [key, update.user_data[key], bot.message.chat.id]
        db_instance.save_params(up_param)
        markup = kb.start_kb()
        bot.message.reply_text(f"Новое значение: {update.user_data[key]}", reply_markup=markup)

        if update.user_data:
            del update.user_data[key]

        return CHOOSING
    except Exception as e:
        markup = kb.params_kb()

        bot.message.reply_text(f"Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз, либо нажмите Back", reply_markup=markup)
        return NEW_VALUE


def start_talk(bot, update):
    bot.message.reply_text('Вот сейчас можно уже ;)')


def clear_data(bot, update):
    kb = KeyBoardGPT()
    markup = kb.start_kb()
    if db_instance.clear_dialogue() == True:
        bot.message.reply_text('Все диалоги удалены из базы.', reply_markup=markup)
    else:
        bot.message.reply_text('Что-то пошло не так...', reply_markup=markup)
    return CHOOSING


#-------------------gpt2 speaking block---------------------------
def echo(bot, update):
    params = db_instance.get_params(bot.message.chat.id)
    answer = model.get_response(bot.message.text, params, db_instance)

    kb = KeyBoardGPT()
    markup = kb.start_kb()
    bot.message.reply_text(answer, reply_markup=markup)

    return CHOOSING


#--------------------transfer block-----------------------
def style_transfer_menu(bot, update):
    kb = KeyBoardGPT()
    markup = kb.style_kb()
    bot.message.reply_text("Style Transfer Menu", reply_markup=markup)

    return STYLE_CONT_CHOOSE


#-----------------style params block-----------------------
def style_params_main(bot, update):
    kb = KeyBoardGPT()
    markup = kb.params_style_main_kb()
    bot.message.reply_text("Style Parameters Main Menu", reply_markup=markup)

    return STYLE_PARAM


#-------------style params model block--------------------
def style_params_model(bot, update):
    db_style = DBStyle_Transfer(bot.message.chat.id)
    kb = KeyBoardGPT()
    markup = kb.params_style_kb()

    update.user_data.clear()

    style_params = db_style.get_base_vgg_params(bot.message.chat.id)
    bot.message.reply_text(f'На данный момент VGG19 имеет такие базовые параметры: \n' +
            f'epoch numbers(steps): {style_params["epoch numbers"]}\nshow cost every: {style_params["show cost [steps]"]} steps\n' +
            f'device: {str(style_params["device[0:cpu, 1:cuda]"])}[0:cpu, 1:cuda] (Only cpu now)\nimage_size: {style_params["image size"]}')

    bot.message.reply_text('Что будем менять?', reply_markup=markup)

    return TYPING_REPLY


#--------------change base vgg params block------------
def received_style_param_information(bot, update):
    db_style = DBStyle_Transfer(bot.message.chat.id)
    user_data = update.user_data
    text = bot.message.text

    update.user_data['choice'] = text
    category = user_data['choice']

    user_data[category] = text
    del user_data['choice']

    params = db_style.get_base_vgg_params(bot.message.chat.id)
    bot.message.reply_text(f"Старое значение: {str(params[text])}")

    kb = KeyBoardGPT()
    markup = kb.params_style_kb()
    bot.message.reply_text("Введите новое значение: ", reply_markup=markup)

    return NEW_STYLE_VALUE


def changing_style_base_params(bot, update):

    db_style = DBStyle_Transfer(bot.message.chat.id)
    kb = KeyBoardGPT()
    try:
        key = [val for val in update.user_data.keys()][0]
        text = bot.message.text
        update.user_data[key] = text

        up_param = [key, update.user_data[key], bot.message.chat.id]
        markup = kb.params_style_main_kb()

        if up_param[0] == 'device[0:cpu, 1:cuda]':
            bot.message.reply_text(f"Не, сорри, у нас только cpu. Тут как-бы без вариантов", reply_markup=markup)
            if update.user_data:
                del update.user_data[key]
            return STYLE_CONT_CHOOSE

        db_style.save_style_params(up_param)
        bot.message.reply_text(f"Новое значение: {update.user_data[key]}", reply_markup=markup)

        if update.user_data:
            del update.user_data[key]

        return STYLE_CONT_CHOOSE
    except Exception as e:
        markup = kb.params_style_kb()

        bot.message.reply_text(f"Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз, либо нажмите Back", reply_markup=markup)
        return NEW_STYLE_VALUE


#----------------params VGG19 layers block---------------
def style_params_layers(bot, update):
    kb = KeyBoardGPT()
    markup = kb.params_st_ct_layers_kb()
    update.user_data.clear()

    bot.message.reply_text(f'Тут можно назначить Conv2d слои для content и style изображения.\n'
                           f'0 - Это значит, что слой вообще не участвует в Style Transfer\n'
                           f'1(или любая другая цифра) - Слой активен.')

    bot.message.reply_text('Чё куда?', reply_markup=markup)

    return TYPING_REPLY


#--------------params VGG19 content layers---------------------
def content_layers(bot, update):
    kb = KeyBoardGPT()
    markup = kb.params_layers_all_kb()

    bot.message.reply_text(f'Тут можно назначить Conv2d слои для CONTENT изображения.\n'
                           f'0 - Это значит, что слой вообще не участвует в Style Transfer\n'
                           f'1(или любая другая цифра) - Слой активен.')

    db_style = DBStyle_Transfer(bot.message.chat.id)
    con_list = db_style.get_cl_values()

    bot.message.reply_text(f'Текущие значения для Content Image следующие:\n'
                           f'conv1_1(3, 64): {con_list[0]}\nconv1_2(64, 64): {con_list[1]}\n'
                           f'conv2_1(64,128): {con_list[2]}\nconv2_2(128,128): {con_list[3]}\n'
                           f'conv3_1(128,256): {con_list[4]}\nconv3_2(256,256): {con_list[5]}\n'
                           f'conv3_3(256,256): {con_list[6]}\nconv3_4(256,256): {con_list[7]}\n'
                           f'conv4_1(256,512): {con_list[8]}\nconv4_2(512,512): {con_list[9]}\n'
                           f'conv4_3(512,512): {con_list[10]}\nconv4_4(512,512): {con_list[11]}\n'
                           f'conv5_1(512,512): {con_list[12]}\nconv5_2(512,512): {con_list[13]}\n'
                           f'conv5_3(512,512): {con_list[14]}\nconv5_4(512,512): {con_list[15]}')

    bot.message.reply_text('Что-то менять будем?', reply_markup=markup)

    return CONTENT_LAYERS_CHOICE


#-----------change content layers params-------------------
def received_cl_param_information(bot, update):
    db_style = DBStyle_Transfer(bot.message.chat.id)
    user_data = update.user_data
    text = bot.message.text

    update.user_data['choice'] = text
    category = user_data['choice']

    user_data[category] = text
    del user_data['choice']

    cl_list = db_style.get_cl_values()
    params = {}
    for idx, val in enumerate(cl_list):
        params[layers_dict[str(idx)]] = val

    bot.message.reply_text(f"Старое значение: {str(params[text])}")

    kb = KeyBoardGPT()
    markup = kb.params_layers_all_kb()
    bot.message.reply_text("Введите новое значение: ", reply_markup=markup)

    return CONTENT_LAYERS_CHOICE


def changing_cl_params(bot, update):
    db_style = DBStyle_Transfer(bot.message.chat.id)
    kb = KeyBoardGPT()
    try:
        key = [val for val in update.user_data.keys()][0]
        text = bot.message.text
        update.user_data[key] = text

        up_param = [key, update.user_data[key], bot.message.chat.id]
        markup = kb.params_layers_all_kb()

        db_style.save_cl_params(up_param)
        bot.message.reply_text(f"Новое значение: {update.user_data[key]}", reply_markup=markup)

        if update.user_data:
            del update.user_data[key]

        return CONTENT_LAYERS_CHOICE
    except Exception as e:
        markup = kb.params_style_kb()

        bot.message.reply_text(f"Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз, либо нажмите Back", reply_markup=markup)
        return CONTENT_LAYERS_CHOICE


#--------------params VGG19 style layers---------------------
def style_layers(bot, update):

    kb = KeyBoardGPT()
    markup = kb.params_layers_all_kb()

    bot.message.reply_text(f'Тут можно назначить Conv2d слои для STYLE изображения.\n'
                           f'0 - Это значит, что слой вообще не участвует в Style Transfer\n'
                           f'1(или любая другая цифра) - Слой активен.')

    db_style = DBStyle_Transfer(bot.message.chat.id)
    stl_list = db_style.get_sl_values()

    bot.message.reply_text(f'Текущие значения для STYLE Image следующие:\n'
                           f'conv1_1(3, 64): {stl_list[0]}\nconv1_2(64, 64): {stl_list[1]}\n'
                           f'conv2_1(64,128): {stl_list[2]}\nconv2_2(128,128): {stl_list[3]}\n'
                           f'conv3_1(128,256): {stl_list[4]}\nconv3_2(256,256): {stl_list[5]}\n'
                           f'conv3_3(256,256): {stl_list[6]}\nconv3_4(256,256): {stl_list[7]}\n'
                           f'conv4_1(256,512): {stl_list[8]}\nconv4_2(512,512): {stl_list[9]}\n'
                           f'conv4_3(512,512): {stl_list[10]}\nconv4_4(512,512): {stl_list[11]}\n'
                           f'conv5_1(512,512): {stl_list[12]}\nconv5_2(512,512): {stl_list[13]}\n'
                           f'conv5_3(512,512): {stl_list[14]}\nconv5_4(512,512): {stl_list[15]}')

    bot.message.reply_text('Что-то менять будем?', reply_markup=markup)

    return STYLE_LAYERS_CHOICE


#-----------change style layers params-------------------
def received_sl_param_information(bot, update):
    db_style = DBStyle_Transfer(bot.message.chat.id)
    user_data = update.user_data
    text = bot.message.text

    update.user_data['choice'] = text
    category = user_data['choice']

    user_data[category] = text
    del user_data['choice']

    sl_list = db_style.get_sl_values()

    params = {}
    for idx, val in enumerate(sl_list):
        params[layers_dict[str(idx)]] = val

    bot.message.reply_text(f"Старое значение: {str(params[text])}")

    kb = KeyBoardGPT()
    markup = kb.params_layers_all_kb()
    bot.message.reply_text("Введите новое значение: ", reply_markup=markup)

    return STYLE_LAYERS_CHOICE


def changing_sl_params(bot, update):

    db_style = DBStyle_Transfer(bot.message.chat.id)
    kb = KeyBoardGPT()
    try:
        key = [val for val in update.user_data.keys()][0]
        text = bot.message.text
        update.user_data[key] = text

        up_param = [key, update.user_data[key], bot.message.chat.id]
        markup = kb.params_layers_all_kb()

        db_style.save_sl_params(up_param)
        bot.message.reply_text(f"Новое значение: {update.user_data[key]}", reply_markup=markup)

        if update.user_data:
            del update.user_data[key]

        return STYLE_CONT_CHOOSE
    except Exception as e:
        markup = kb.params_style_kb()

        bot.message.reply_text(f"Ошибка ввода! ")
        bot.message.reply_text(f"Введите значение еще раз либо нажмите Back", reply_markup=markup)
        return STYLE_LAYERS_CHOICE


#------------reset layer parameters---------------------
def reset_layers(bot, update):
    db_style = DBStyle_Transfer(bot.message.chat.id)
    kb = KeyBoardGPT()

    base_style = VggModel.base_style_layer
    base_cont = VggModel.base_cont_layer
    markup = kb.params_st_ct_layers_kb()
    if db_style.reset_layers(base_style, base_cont):

        bot.message.reply_text("Слои VGG в исходном состоянии.", reply_markup=markup)
    else:
        bot.message.reply_text("Что-то пошло не так.", reply_markup=markup)
    return TYPING_REPLY


#-----------------style image upload------------------------
def style_image(bot, update):
    kb = KeyBoardGPT()
    markup = kb.back_kb()
    bot.message.reply_text("Upload style image, please", reply_markup=markup)

    return UPLOAD_STYLE


def upload_style_image(bot, update):
    kb = KeyBoardGPT()
    markup = kb.back_kb()

    db_uploader = DBStyle_Transfer(bot.message.chat.id)

    file = update.bot.get_file(bot.message.photo[-1].file_id)
    bot.message.reply_text("We Got style file. To upload content image  or make style transfer - press 'Back'", reply_markup=markup)

    if db_uploader.save_file_link(file_path=file['file_path'], is_style=True):
        return UPLOAD_STYLE
    else:
        kb = KeyBoardGPT()
        markup = kb.style_kb()
        bot.message.reply_text("Something wrong. Try again", reply_markup=markup)

        return STYLE_CONT_CHOOSE


#-----------------content image upload------------------------
def content_image(bot, update):
    kb = KeyBoardGPT()
    markup = kb.back_kb()
    bot.message.reply_text("Upload content image, please",reply_markup=markup)

    return UPLOAD_CONTENT


def upload_content_image(bot, update):
    kb = KeyBoardGPT()
    markup = kb.back_kb()
    db_uploader = DBStyle_Transfer(bot.message.chat.id)

    file = update.bot.get_file(bot.message.photo[-1].file_id)
    bot.message.reply_text("We Got content file. To upload style image or make style transfer - press 'Back'.",reply_markup=markup)

    if db_uploader.save_file_link(file_path=file['file_path'], is_style=False):
        return UPLOAD_CONTENT
    else:
        kb = KeyBoardGPT()
        markup = kb.style_kb()
        bot.message.reply_text("Something wrong. Try again", reply_markup=markup)

        return STYLE_CONT_CHOOSE


#-----------------run style transfer------------------------
def run_transfer(bot, update):

    db_style = DBStyle_Transfer(bot.message.chat.id)
    style_params = db_style.get_base_vgg_params(bot.message.chat.id)

    try:
        stile_link = db_style.load_file_link(is_style=True)
        content_link = db_style.load_file_link()

    except Exception as ex:
        kb = KeyBoardGPT()
        markup = kb.style_kb()
        bot.message.reply_text("Something wrong with your image. Try again", reply_markup=markup)
        return UPLOAD_STYLE

    try:
        image_preprocess = ImagePreprocessing(style_params["image size"])
        style = image_preprocess.image_loader(file_path=stile_link)
        content = image_preprocess.image_loader(file_path=content_link)
        input_img = content
    except Exception as ex:
        kb = KeyBoardGPT()
        markup = kb.style_kb()
        bot.message.reply_text("Something wrong with your image processing. Try again", reply_markup=markup)
        return UPLOAD_STYLE

    try:
        # #pre-trained on local space!
        # model = Vgg()
        model = VggModel()
        # model.freeze_layers()
        # get content and style features only once before training
        # content_features = model.get_features(content)
        # style_features = model.get_features(style)

        bot.message.reply_text("Wait.... it's take a time!")

        #style-transfer model v2
        # img = model.run(content=content, content_features=content_features, style_features=style_features,
        #                 steps=style_params["epoch numbers"], show_every=style_params["show cost [steps]"])

        stl_layers = db_style.get_sl_values()
        con_layers = db_style.get_cl_values()
        output = model.run_style_transfer(content, style, input_img, num_steps=style_params["epoch numbers"],
                       show_every=style_params["show cost [steps]"], style_layers=stl_layers, content_layers=con_layers)

        img = image_preprocess.im_detorch(output)
        img_byte_arr = image_preprocess.im_to_bytearray(img)

        update.bot.send_photo(chat_id=bot.message.chat.id, photo=img_byte_arr)
        kb = KeyBoardGPT()
        markup = kb.params_style_main_kb()
        bot.message.reply_text("Done! Make your choice.", reply_markup=markup)

        return STYLE_CONT_CHOOSE
    except Exception as ex:
        kb = KeyBoardGPT()
        markup = kb.style_kb()
        bot.message.reply_text("Something wrong with your model. Try again", reply_markup=markup)
        return UPLOAD_STYLE


def done(bot, update):
    user_data = update.user_data
    if 'choice' in user_data:
        del user_data['choice']

    bot.message.reply_text(f"До новых встреч )", reply_markup=kb_remove)
    user_data.clear()
    return CHOOSING


def main():
    updater = Updater(TOKEN)

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING: [
                MessageHandler(Filters.regex('^Just Talk$'), start_talk),
                MessageHandler(Filters.regex('^Params$'), gpt_params),
                MessageHandler(Filters.regex('^Clear Talk Data$'), clear_data),
                MessageHandler(Filters.regex('^Make Style Transfer$'), style_transfer_menu),
                MessageHandler(Filters.regex('^Exit$'), done),
            ],
            TYPING_CHOICE: [
                MessageHandler(Filters.regex('^(max_length|no_repeat_ngram_size|do_sample|top_k|num_return_sequences|' +
                                             'top_p|temperature|device|is_always_use_length|' +
                                             'length_generate)$'), changing_params),

                MessageHandler(Filters.regex('^Back'), main_menu),
            ],
            TYPING_REPLY: [
                MessageHandler(Filters.regex('^(max_length|no_repeat_ngram_size|do_sample|top_k|num_return_sequences|' +
                                             'top_p|temperature|device|is_always_use_length|' +
                                             'length_generate)$'), received_information),

                MessageHandler(Filters.regex('^(epoch numbers|show cost \[steps\]|device\[0:cpu, 1:cuda\]|' +
                                             'image size)$'), received_style_param_information),

                MessageHandler(Filters.regex('^Style layers$'), style_layers),
                MessageHandler(Filters.regex('^Content layers$'), content_layers),

                MessageHandler(Filters.regex('^Back$'), main_menu),
                MessageHandler(Filters.regex('^Main menu$'), main_menu),
                MessageHandler(Filters.regex('^Reset layers$'), reset_layers),
            ],
            CHANGING: [
                MessageHandler(Filters.regex('^(max_length|no_repeat_ngram_size|do_sample|top_k|num_return_sequences|' +
                                             'top_p|temperature|device|is_always_use_length|' +
                                             'length_generate)$'), gpt_params),
            ],
            NEW_VALUE: [
                MessageHandler(Filters.regex('^Back'), main_menu),
                MessageHandler(Filters.regex('^(max_length|no_repeat_ngram_size|do_sample|top_k|num_return_sequences|' +
                                             'top_p|temperature|device|is_always_use_length|' +
                                             'length_generate)$'), gpt_params),
                MessageHandler(Filters.text & ~(Filters.command | Filters.regex('^Done$')), changing_params),
            ],
            NEW_STYLE_VALUE: [
                MessageHandler(Filters.regex('^Back$'), style_params_model),
                MessageHandler(Filters.regex('^Model Params$'), style_params_model),

                MessageHandler(Filters.regex('^(epoch numbers|show cost \[steps\]|device\[0:cpu, 1:cuda\]|' +
                                             'image size)$'), changing_style_base_params),

                MessageHandler(Filters.text & ~(Filters.command | Filters.regex('^Done$')), changing_style_base_params)
            ],
            STYLE_CONT_CHOOSE: [
                MessageHandler(Filters.regex('^Back$'), main_menu),
                MessageHandler(Filters.regex('^Model Params$'), style_params_model),
                MessageHandler(Filters.regex('^upload style image$'), style_image),
                MessageHandler(Filters.regex('^upload content image$'), content_image),
                MessageHandler(Filters.regex('^style params$'), style_params_main),
                MessageHandler(Filters.regex('^make transfer$'), run_transfer),
            ],
            STYLE_PARAM: [
                MessageHandler(Filters.regex('^Back$'), style_transfer_menu),
                MessageHandler(Filters.regex('^Main menu$'), main_menu),
                MessageHandler(Filters.regex('^Model Params$'), style_params_model),
                MessageHandler(Filters.regex('^Layers Vgg$'), style_params_layers),

                MessageHandler(Filters.regex('^Reset layers$'), style_params_model)
            ],
            UPLOAD_STYLE: [
                MessageHandler(Filters.regex('^Back$'), style_transfer_menu),
                MessageHandler(Filters.regex('^Main menu$'), main_menu),

                MessageHandler(Filters.photo, upload_style_image),
            ],
            UPLOAD_CONTENT: [
                MessageHandler(Filters.regex('^Back$'), style_transfer_menu),
                MessageHandler(Filters.regex('^Main menu$'), main_menu),

                MessageHandler(Filters.photo, upload_content_image),
            ],
            CONTENT_LAYERS_CHOICE:[
                MessageHandler(Filters.regex('^Back$'), style_params_layers),

                MessageHandler(Filters.regex('^(conv1_1\(3, 64\)|conv1_2\(64, 64\)|conv2_1\(64,128\)|conv2_2\(128,128\)'+
                                             '|conv3_1\(128,256\)|conv3_2\(256,256\)|conv3_3\(256,256\)|conv3_4\(256,256\)'+
                                             '|conv4_1\(256,512\)|conv4_2\(512,512\)|conv4_3\(512,512\)|conv4_4\(512,512\)'+
                                             '|conv5_1\(512,512\)|conv5_2\(512,512\)|conv5_3\(512,512\)|conv5_4\(512,512\))$'), received_cl_param_information),

                MessageHandler(Filters.text & ~(Filters.command | Filters.regex('^Done$')), changing_cl_params),
            ],
            STYLE_LAYERS_CHOICE: [
                MessageHandler(Filters.regex('^Back$'), style_params_layers),

                MessageHandler(
                    Filters.regex('^(conv1_1\(3, 64\)|conv1_2\(64, 64\)|conv2_1\(64,128\)|conv2_2\(128,128\)' +
                                  '|conv3_1\(128,256\)|conv3_2\(256,256\)|conv3_3\(256,256\)|conv3_4\(256,256\)' +
                                  '|conv4_1\(256,512\)|conv4_2\(512,512\)|conv4_3\(512,512\)|conv4_4\(512,512\)' +
                                  '|conv5_1\(512,512\)|conv5_2\(512,512\)|conv5_3\(512,512\)|conv5_4\(512,512\))$'),
                    received_sl_param_information),

                MessageHandler(Filters.text & ~(Filters.command | Filters.regex('^Done$')), changing_sl_params),
            ],
        },
        fallbacks=[MessageHandler(Filters.regex('^Done$'), done)],
    )

    dispatcher = updater.dispatcher
    dispatcher.add_handler(conv_handler)
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))

    # updater.start_polling()

    # updater.start_webhook(
    #     listen='0.0.0.0',
    #     port=PORT,
    #     url_path=TOKEN,
    #     webhook_url= f'https://{IP_ADDRESS}/{TOKEN}',
    #     cert='cert.pem')
    
    updater.start_webhook(
        listen='0.0.0.0',
        port=PORT,
        url_path=TOKEN,
        key='private.key',
        cert='cert.pem',
        webhook_url=f'https://{IP_ADDRESS}:8443/{TOKEN}')

    updater.idle()


if __name__ == '__main__':
    main()