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
from flask import Flask, render_template, request, jsonify, url_for
from gtts import gTTS

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DURATION = 10
SAVE_PATH = "D:/audioai"
UDP_IP = "0.0.0.0"
UDP_PORT = 1234

os.makedirs(SAVE_PATH, exist_ok=True)

app = Flask(__name__)


# Hàm tiền xử lý văn bản
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


def get_next_filename(prefix="audio"):
    index = 1
    while os.path.exists(os.path.join(SAVE_PATH, f"{prefix}_{index}.wav")):
        index += 1
    return os.path.join(SAVE_PATH, f"{prefix}_{index}.wav")


def receive_audio():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Đang lắng nghe gói UDP trên cổng {UDP_PORT}...")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)
    filename = get_next_filename("audio")
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
        print(f"Hoàn tất thu âm! File đã lưu: {filename}")
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
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    train_data = pd.read_csv('train_data_processed.csv')
    tfidf_matrix = tfidf_vectorizer.transform(train_data['question_processed'])
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tfidf_matrix)

    query_processed = preprocess_text(query, mode='content')
    query_tfidf = tfidf_vectorizer.transform([query_processed])
    scores = cosine_similarity(query_tfidf, tfidf).flatten()
    best_idx = np.argmax(scores)

    matched_question = train_data.iloc[best_idx]['question']
    matched_answer = train_data.iloc[best_idx]['answer']
    matched_link = train_data.iloc[best_idx]['link']
    similarity_score = scores[best_idx]




    # Tạo và phát file âm thanh từ câu trả lời
    tts = gTTS(text=matched_answer, lang='vi', slow=False)
    audio_file = get_next_filename("answer").replace(".wav", ".mp3")
    tts.save(audio_file)

    # Phát file âm thanh ngay lập tức
    os.system(f"start {audio_file}")  # Windows

    return {
        'input_question': query,
        'matched_question': matched_question,
        'answer': matched_answer,
        'link': matched_link,
        'similarity_score': similarity_score,
        'audio_url': audio_file
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/record', methods=['POST'])
def record():
    try:
        audio_file = receive_audio()
        if audio_file:
            text_result = audio_to_text(audio_file)
            if text_result and "Không thể nhận diện" not in text_result and "Lỗi khi kết nối" not in text_result:
                prediction = predict_answer(text_result)
                print(prediction['similarity_score']);
                return jsonify({
                    'status': 'success',
                    'input_question': prediction['input_question'],
                    'matched_question': prediction['matched_question'],
                    'answer': prediction['answer'],
                    'link': prediction['link'],
                    'similarity_score': prediction['similarity_score'],
                    'audio_url': url_for('static', filename=os.path.basename(prediction['audio_url']))
                })
            else:
                return jsonify({'status': 'error', 'message': text_result})
        else:
            return jsonify({'status': 'error', 'message': 'Không thu được âm thanh'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Đã xảy ra lỗi: {str(e)}'})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)