import re
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import torch


INPUT_VOCAB_SIZE = 52381
OUTPUT_VOCAB_SIZE = 41061
INPUT_ENCODER_LENGTH = 17
INPUT_DECODER_LENGTH = 29
def input_processor(input_sentence,tokenizer, pad_seq): #chuyển câu đầu vào và chuyển thành index, thêm padding cần thiết để đưa vào encoder (phục vụ cho test model)

  encoder_input = preprocess(input_sentence, add_start_token= True, add_end_token=True)

  tokenized_text = tokenizer.texts_to_sequences([encoder_input])
  if pad_seq == True:
    tokenized_text = pad_sequences(tokenized_text, maxlen=INPUT_ENCODER_LENGTH, padding="post")

  tokenized_text = tf.convert_to_tensor(tokenized_text, dtype = tf.float32)
  return tokenized_text

def preprocess(t, add_start_token, add_end_token):

  if add_start_token == True and add_end_token == False: #nếu thêm start token và không thêm end token
    t = '<start>'+' '+t
  if add_start_token == False and add_end_token == True: #nếu không thêm start token và thêm end token
    t = t+' '+'<end>'
  if add_start_token == True and add_end_token == True: #nếu thêm cả start token và end token
    t = '<start>'+' '+t+' '+'<end>'

  t = re.sub(' +', ' ', t) #loại bỏ khoảng trắng thừa
  return t


def predict_sentence(model,input_sentence,link_to_final_dataset, max_length=29, device="cuda"):
    data = pd.read_csv(link_to_final_dataset)
    encoder_input = [preprocess(line, add_start_token= True, add_end_token=True) for line in data['error']]
    decoder_input = [preprocess(line, add_start_token= True, add_end_token=False) for line in data['correct']]
    decoder_output = [preprocess(line, add_start_token= False, add_end_token=True) for line in data['correct']]
    tokenizer = Tokenizer(filters='', split=" ")
    tokenizer.fit_on_texts(encoder_input)

    decoder_data = decoder_input.copy() #copy decoder_input sang decoder_data
    decoder_data.extend(decoder_output)
    word_index = tokenizer.word_index
    out_tokenizer = Tokenizer(filters='', split=" ") #tạo tokenizer cho decoder_data
    out_tokenizer.fit_on_texts(decoder_data) #fit dữ liệu vào tokenizer
    word_index = out_tokenizer.word_index
    model.eval()  # Chuyển model sang chế độ eval

    start_token_in = 1  # Start token của tokenizer (input)
    end_token_in = 2    # End token của tokenizer (input)
    start_token_out = 3  # Start token của out_tokenizer (output)
    end_token_out = 4    # End token của out_tokenizer (output)

    # 1️⃣ Xử lý đầu vào (encoder_input)
    input_tensor = input_processor(input_sentence, tokenizer, pad_seq=True)  # Đưa vào hàm tiền xử lý có sẵn
    input_tensor = torch.tensor(input_tensor.numpy(), dtype=torch.long).to(device)  # (1, 17)

    # 2️⃣ Khởi tạo decoder_input với start token
    decoder_input = torch.tensor([[start_token_out]], dtype=torch.long).to(device)  # (1,1)

    predicted_sentence = []

    with torch.no_grad():  # Không cần tính gradient khi inference
        for _ in range(max_length):  # Giới hạn tối đa 29 từ
            # 3️⃣ Dự đoán từ tiếp theo
            output = model([input_tensor, decoder_input])  # (1, seq_len, vocab_size)

            # 4️⃣ Chọn từ có xác suất cao nhất (Greedy Search)
            next_word = torch.argmax(output[:, -1, :], dim=-1).item()

            # 5️⃣ Nếu gặp token <end> thì dừng lại
            if next_word == end_token_out:
                break

            # 6️⃣ Thêm từ vào kết quả
            predicted_sentence.append(next_word)

            # 7️⃣ Cập nhật decoder_input để tiếp tục sinh từ mới
            next_word_tensor = torch.tensor([[next_word]], dtype=torch.long).to(device)  # (1,1)
            decoder_input = torch.cat([decoder_input, next_word_tensor], dim=1)  # Nối thêm từ vào decoder_input

            # 8️⃣ Nếu đã đạt độ dài tối đa 29 thì dừng
            if decoder_input.shape[1] >= max_length:
                break

    # 9️⃣ Chuyển index thành câu hoàn chỉnh sử dụng out_tokenizer
    predicted_words = [out_tokenizer.index_word.get(idx, "<unk>") for idx in predicted_sentence]
    
    return " ".join(predicted_words)