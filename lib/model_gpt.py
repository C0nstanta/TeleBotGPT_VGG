from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os





params = {
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


class Model:
    def __init__(self):
        my_dir = os.getcwd()
        self.tokenizer = AutoTokenizer.from_pretrained(my_dir + "/model_gpt/")
        self.model = AutoModelForCausalLM.from_pretrained(my_dir + "/model_gpt/")
        self.step = 0


    def get_response(self, chat_id, user_answer, params, db_gpt):
        self.conversation_text = []

        result = db_gpt.get_dialogue(chat_id)
        if result:
            self.conversation_text.extend(result)

        params = self.compile_params(params)
        self.conversation_text.append({"speaker": 0, "text": user_answer})#

        input_ids = self.get_input_ids(params)
        qpt_answer = self.generate_answer(input_ids, params)
        self.conversation_text.append({"speaker": 1, "text": qpt_answer})

        print(self.conversation_text)
        db_gpt.rewrite_dialogue(chat_id, self.conversation_text)

        return qpt_answer


    def get_input_ids(self, params):
        input_text = ''
        conv_text_tmp = self.conversation_text if len(self.conversation_text) < 7 else self.conversation_text[-7:]
        for input in conv_text_tmp:
            if params['is_always_use_length']:
                length_rep = len(self.tokenizer.encode(input['text']))
                if length_rep <= 15:
                    length_param = '1'
                elif length_rep <= 50:
                    length_param = '2'
                elif length_rep <= 256:
                    length_param = '3'
                else:
                    length_param = '-'
            else:
                length_param = '-'
            input_text += f"|{input['speaker']}|{length_param}|{input['text']}"
        input_text += f"|1|{params['length_generate']}|"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')

        return input_ids


    def generate_answer(self, input_ids, params):
        params_upd = {
            'max_length': params['max_length'],
            'no_repeat_ngram_size': params['no_repeat_ngram_size'],
            'do_sample': params['do_sample'],
            'top_k': params['top_k'],
            'top_p': params['top_p'],
            'temperature': params['temperature'],
            'num_return_sequences': params['num_return_sequences'],
            'device': params['device'],
            'mask_token_id': self.tokenizer.mask_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }

        output_ids = self.model.generate(input_ids, **params_upd)
        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output = outputs.split("|")[-1]

        return output


    def compile_params(self, params):
        params['max_length'] = int(params['max_length'])
        params['no_repeat_ngram_size'] = int(params['no_repeat_ngram_size'])
        params['do_sample'] = bool(params['do_sample'])
        params['top_k'] = int(params['top_k'])
        params['top_p'] = float(params['top_p'])
        params['temperature'] = float(params['temperature'])
        params['num_return_sequences'] = int(params['num_return_sequences'])
        params['device'] = int(params['device'])
        params['is_always_use_length'] = bool(params['is_always_use_length'])
        params['length_generate'] = int(params['length_generate'])
        return params


    def tokken_padding(self, input_ids):
        token_np = np.array(input_ids)
        len_np = token_np.shape[1]

        delta = 256 - len_np
        if delta >= 0:
            token_pad = np.append(np.zeros((1, delta)), token_np)
        else:
            delta2 = len_np - 256
            token_pad = token_np[:, delta2:len_np]
        token_pad = token_pad.reshape(-1, 256)
        return token_pad.astype('int32')


def get_model():
    return model


model = Model()