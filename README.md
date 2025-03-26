# NLP
 This project is under the course NLP

Explain the file:
1. Train the model:
- Building LSTM model: code to train the model using LSTM as encoder and decoder with the torch framework
2. Test the model:
- requirements.txt: các thư viện cần tải để test model
- dataset_for_tokenize.csv: the dataset for model's tokenizer
- in_embedding.npy: tệp numpy lưu trữ ma trận embedding cho đầu vào ([download tại đây](https://drive.google.com/file/d/1xpQlW4kb56hO5-UEUVyiURG40D8kt-7G/view?usp=sharing)).
- out_embedding.npy: tệp numpy lưu trữ ma trận embedding cho đầu ra ([download tại đây](https://drive.google.com/file/d/1qZEIiA3EYlZvA-opbYLh7iSjXgY-53xb/view?usp=sharing)).
- best_model.pth: lưu tham số của model đã được train ([download tại đây](https://drive.google.com/file/d/19X9ydtJ764lwZI-w-huzT84aNfc81UFK/view?usp=sharing)).
- create_model.py: tạo cấu trúc model
- data_preprocess: xử lý câu đầu vào để sẵn sàng đưa vào model 
- predict_new: file chính, dùng để đoán đầu ra của model
  
Quy trình test model:
1. Bước 1: Download tất cả các file cần thiết của mục 2
2. Bước 2: chạy pip install path_to_requirements trong cmd
3. Bước 3: vào file predict_new, sửa các link cho đúng và câu muốn sửa lỗi
4. Bước 4: chạy file predict_new để nhận dự đoán từ model
