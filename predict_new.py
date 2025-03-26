from create_model import create_model
import torch
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
from data_preprocess import input_processor, preprocess, predict_sentence
link_to_tokenize_dataset = r"C:\Users\Admin\Downloads\NLP\Model_NLP\Testmodel\dataset_for_tokenize.csv" #chủ động thay đổi đường dẫn tới file
link_to_in_embedding = r"C:\Users\Admin\Downloads\NLP\Model_NLP\Testmodel\in_embedding.npy" #chủ động thay đổi đường dẫn tới file
link_to_out_embedding = r"C:\Users\Admin\Downloads\NLP\Model_NLP\Testmodel\out_embedding.npy" #chủ động thay đổi đường dẫn tới file
model = create_model(link_to_in_embedding,link_to_out_embedding) #load model đã train
model.load_state_dict(torch.load(r"C:\Users\Admin\Downloads\NLP\Model_NLP\Testmodel\best_model.pth")) #chủ động thay đổi đường dẫn tới file
input_sentence = "i love you so muchhh ."
output_sentence = predict_sentence(model,input_sentence,link_to_tokenize_dataset)
print("Câu đã sửa:", output_sentence)
