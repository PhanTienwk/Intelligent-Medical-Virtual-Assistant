import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import pickle



def preprocess_text(text, mode='content'):
    if mode == 'content':
        text = re.sub(r'<[^>]+>', '', text)
        text = text.lower()
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = text.replace('\n', ' ')
        text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_.1234567890:,?]', ' ',
                      text)
        text = re.sub(r'(?<=[.!?])[^.!?]*\?', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.{2,3}', '.', text)
    else:
        text = text.lower()
        text = text.replace('.', ',')
    return text



data = pd.read_csv('train.csv')
data = data.dropna(subset=['question', 'answer', 'link'])
data['question_processed'] = data['question'].apply(lambda x: preprocess_text(x, mode='content'))


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['question_processed'])
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(tfidf_matrix)


answer_vectorizer = TfidfVectorizer()
answer_tfidf = answer_vectorizer.fit_transform(train_data['answer'])


# Hàm dự đoán câu trả lời
def predict_answer(query):
    query_processed = preprocess_text(query, mode='content')
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    scores = cosine_similarity(query_tfidf, tfidf).flatten()
    best_idx = np.argmax(scores)
    return train_data.iloc[best_idx]['answer'], train_data.iloc[best_idx]['link']


# Hàm tính độ tương đồng giữa hai câu trả lời
def compute_answer_similarity(pred_answer, actual_answer):
    pred_tfidf = answer_vectorizer.transform([pred_answer])
    actual_tfidf = answer_vectorizer.transform([actual_answer])
    similarity = cosine_similarity(pred_tfidf, actual_tfidf)[0][0]
    return similarity


# Đánh giá mô hình
def evaluate_model():
    correct = 0
    total_similarity = 0
    print("\nĐánh giá trên tập kiểm tra:")
    for _, row in test_data.iterrows():
        pred_answer, pred_link = predict_answer(row['question'])
        is_correct = (pred_answer == row['answer']) and (pred_link == row['link'])
        similarity = compute_answer_similarity(pred_answer, row['answer'])
        total_similarity += similarity
        correct += is_correct

        print(f"Question: {row['question']}")
        print(f"Predicted Answer: {pred_answer}")
        print(f"Actual Answer: {row['answer']}")
        print(f"Similarity Score: {similarity:.4f}")
        print(f"Correct: {is_correct}")
        print("-" * 50)

    accuracy = (correct / len(test_data)) * 100
    avg_similarity = total_similarity / len(test_data)
    print(f"Độ chính xác (Precision@1): {accuracy:.2f}%")
    print(f"Độ tương đồng trung bình: {avg_similarity:.4f}")
    return accuracy, avg_similarity


# Chạy đánh giá
accuracy, avg_similarity = evaluate_model()


with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('answer_vectorizer.pkl', 'wb') as f:
    pickle.dump(answer_vectorizer, f)
train_data.to_csv('train_data_processed.csv', index=False)
print("Đã lưu mô hình và dữ liệu!")

# print("\n=== Kiểm tra mô hình bằng cách nhập câu hỏi ===")
# print("Nhập 'exit' để thoát.")
# while True:
#     user_input = input("Nhập câu hỏi của bạn: ").strip()
#     if user_input.lower() == 'exit':
#         print("Đã thoát chương trình.")
#         break
#     if not user_input:
#         print("Vui lòng nhập câu hỏi!")
#         continue
#
#     answer, link = predict_answer(user_input)
#     print(f"Question: {user_input}")
#     print(f"Answer: {answer}")
#     print(f"Link: {link}")
#     print("-" * 50)

print("\n=== Kiểm tra mô hình bằng cách nhập câu hỏi ===")
print("Nhập 'exit' để thoát.")
while True:
    user_input = input("Nhập câu hỏi của bạn: ").strip()
    if user_input.lower() == 'exit':
        print("Đã thoát chương trình.")
        break
    if not user_input:
        print("Vui lòng nhập câu hỏi!")
        continue

    # Tiền xử lý và vector hóa câu hỏi nhập vào
    query_processed = preprocess_text(user_input, mode='content')
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    scores = cosine_similarity(query_tfidf, tfidf).flatten()
    best_idx = np.argmax(scores)

    # Lấy dữ liệu từ câu hỏi khớp nhất trong train_data
    matched_question = train_data.iloc[best_idx]['question']
    matched_answer = train_data.iloc[best_idx]['answer']
    matched_link = train_data.iloc[best_idx]['link']

    # Tính độ tương đồng giữa câu hỏi bạn nhập và câu hỏi được match
    matched_question_processed = preprocess_text(matched_question, mode='content')
    matched_question_tfidf = tfidf_vectorizer.transform([matched_question_processed])
    similarity_score = cosine_similarity(query_tfidf, matched_question_tfidf)[0][0]

    print(f"\nCâu hỏi của bạn     : {user_input}")
    print(f"Câu hỏi được tìm thấy: {matched_question}")
    print(f"Độ tương đồng giữa 2 câu hỏi: {similarity_score:.4f}")
    print(f"Câu trả lời dự đoán : {matched_answer}")
    print(f"Link: {matched_link}")
    print("-" * 50)