import speech_recognition as sr

def audio_to_text(audio_file_path):
    # Khởi tạo Recognizer
    recognizer = sr.Recognizer()

    # Đọc file audio
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    try:
        # Chuyển audio thành text tiếng Việt bằng Google Speech Recognition
        text = recognizer.recognize_google(audio_data, language="vi-VN")
        return text
    except sr.UnknownValueError:
        return "Không thể nhận diện được nội dung audio"
    except sr.RequestError as e:
        return f"Lỗi khi kết nối đến dịch vụ: {e}"


# Sử dụng hàm
if __name__ == "__main__":
    # Thay bằng đường dẫn file audio của bạn (phải là WAV)
    audio_path = "audio_20.wav"

    try:
        result = audio_to_text(audio_path)
        print("Kết quả chuyển đổi: ", result)
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")