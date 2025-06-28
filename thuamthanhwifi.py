import socket
import numpy as np
import pyaudio
import wave
import os
from time import time

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
DURATION = 10
SAVE_PATH = "D:/audioai"
UDP_IP = "0.0.0.0"
UDP_PORT = 1234

os.makedirs(SAVE_PATH, exist_ok=True)

def get_next_filename():
    index = 1
    while os.path.exists(os.path.join(SAVE_PATH, f"audio_{index}.wav")):
        index += 1
    return os.path.join(SAVE_PATH, f"audio_{index}.wav")

def receive_audio():
    # Tạo socket UDP với buffer lớn
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)  # Buffer 64KB
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP packets on port {UDP_PORT}...")

    # Khởi tạo PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, output=True)

    # Mở file WAV
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
            data, addr = sock.recvfrom(CHUNK_SIZE * 2 * 4)  # Nhận tối đa 8192 byte
            buffer.extend(data)

            # Xử lý khi đủ 2048 byte
            while len(buffer) >= CHUNK_SIZE * 2:
                audio_data = buffer[:CHUNK_SIZE * 2]
                buffer = buffer[CHUNK_SIZE * 2:]

                samples = np.frombuffer(audio_data, dtype=np.int16)
                stream.write(samples.tobytes())
                wf.writeframes(audio_data)
                chunk_count += 1

                # Cập nhật tiến trình
                if chunk_count % 10 == 0:
                    elapsed = time() - start_time
                    print(f"Processed {chunk_count}/{max_chunks} chunks ({elapsed:.1f}s)")

            # Debug dữ liệu nhận được
            if len(data) != CHUNK_SIZE * 2:
                print(f"Received {len(data)} bytes from {addr}, expected {CHUNK_SIZE * 2}")

        print(f"🎙️ Recording finished! File saved: {filename}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sock.close()
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()

if __name__ == "__main__":
    receive_audio()