import socket
import numpy as np
import pyaudio
import wave
import os
from time import time
import speech_recognition as sr
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DURATION = 10
SAVE_PATH = "D:/audioai"
UDP_IP = "0.0.0.0"
UDP_PORT = 1234

os.makedirs(SAVE_PATH, exist_ok=True)


# Hàm tiền xử lý văn bản cho TF-IDF
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


def get_next_filename():
    """Tạo tên file mới không trùng lặp"""
    index = 1
    while os.path.exists(os.path.join(SAVE_PATH, f"audio_{index}.wav")):
        index += 1
    return os.path.join(SAVE_PATH, f"audio_{index}.wav")


def receive_audio():
    """Thu âm thanh qua UDP và lưu vào file WAV"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Đang lắng nghe gói UDP trên cổng {UDP_PORT}...")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    filename = get_next_filename()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)

    max_chunks = int((SAMPLE_RATE * DURATION) / CHUNK_SIZE)
    chunk_count = 0
    buffer = bytearray()
    start_time = time()

    try:
        while chunk_count < max_chunks:
            data, addr = sock.recvfrom(CHUNK_SIZE * 2 * 4)
            buffer.extend(data)
            while len(buffer) >= CHUNK_SIZE * 2:
                audio_data = buffer[:CHUNK_SIZE * 2]
                buffer = buffer[CHUNK_SIZE * 2:]
                samples = np.frombuffer(audio_data, dtype=np.int16)
                stream.write(samples.tobytes())
                wf.writeframes(audio_data)
                chunk_count += 1
                if chunk_count % 10 == 0:
                    elapsed = time() - start_time
                    print(f"Đã xử lý {chunk_count}/{max_chunks} phần ({elapsed:.1f}s)")
        print(f"🎙️ Hoàn tất thu âm! File đã lưu: {filename}")
        return filename
    except Exception as e:
        print(f"Lỗi: {e}")
        return None
    finally:
        sock.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()


def audio_to_text(audio_file_path):
    """Chuyển âm thanh sang văn bản tiếng Việt"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language="vi-VN")
        return text
    except sr.UnknownValueError:
        return "Không thể nhận diện được nội dung audio"
    except sr.RequestError as e:
        return f"Lỗi khi kết nối đến dịch vụ: {e}"


def predict_answer(query):
    """Dự đoán câu trả lời dựa trên TF-IDF"""
    # Load mô hình và dữ liệu
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    train_data = pd.read_csv('train_data_processed.csv')
    tfidf_matrix = tfidf_vectorizer.transform(train_data['question_processed'])
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tfidf_matrix)

    # Tiền xử lý và vector hóa câu hỏi
    query_processed = preprocess_text(query, mode='content')
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    scores = cosine_similarity(query_tfidf, tfidf).flatten()
    best_idx = np.argmax(scores)

    # Lấy kết quả
    matched_question = train_data.iloc[best_idx]['question']
    matched_answer = train_data.iloc[best_idx]['answer']
    matched_link = train_data.iloc[best_idx]['link']
    similarity_score = scores[best_idx]

    return {
        'input_question': query,
        'matched_question': matched_question,
        'answer': matched_answer,
        'link': matched_link,
        'similarity_score': similarity_score
    }


def main():
    """Hàm chính kết hợp thu âm, chuyển text và dự đoán câu trả lời"""
    try:
        # Thu âm thanh
        audio_file = receive_audio()

        if audio_file:
            print(f"Đang chuyển file {audio_file} sang văn bản...")
            # Chuyển âm thanh sang text
            text_result = audio_to_text(audio_file)
            print("Văn bản từ âm thanh: ", text_result)

            # Nếu chuyển đổi thành công, đưa vào model dự đoán
            if text_result and "Không thể nhận diện" not in text_result and "Lỗi khi kết nối" not in text_result:
                print("\nĐang dự đoán câu trả lời...")
                prediction = predict_answer(text_result)

                print(f"\nCâu hỏi từ âm thanh : {prediction['input_question']}")
                print(f"Câu hỏi khớp nhất  : {prediction['matched_question']}")
                print(f"Độ tương đồng       : {prediction['similarity_score']:.4f}")
                print(f"Câu trả lời         : {prediction['answer']}")
                print(f"Link                : {prediction['link']}")
                print("-" * 50)
            else:
                print("Không thể dự đoán câu trả lời do lỗi chuyển đổi âm thanh.")
        else:
            print("Không có file âm thanh để xử lý.")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")


if __name__ == "__main__":
    main()