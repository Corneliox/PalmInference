from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import threading
import time

app = Flask(__name__)
model = YOLO("palm.pt")

# Safe camera initialization
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Kamera tidak dapat dibuka. Pastikan tidak digunakan oleh aplikasi lain.")

# Biarkan kamera warm-up dulu (opsional tapi direkomendasikan)
# time.sleep(2)

last_frame = None
frame_lock = threading.Lock()

def interpret_lines(label, x, y, w, h, confidence):
    if label == "brain":
        if w > 0.4:
            return "🚂 Alur hidupmu lurus kayak rel kereta... tapi kadang disabotase sinyal"
        else:
            return "📚 Hidupmu seperti skripsi: ditunda-tunda tapi akhirnya kelar juga."

    elif label == "feel":
        if h > 0.3:
            return "💘 Kamu tipe bucin yang suka bilang 'terserah' tapi ngambek diam-diam"
        else:
            return "🔕 Cinta? Apa itu? Kayaknya kamu udah LDR sama jodoh sejak lahir."
    elif label == "feel" and w > 0.6:
        return "Waduh... bisa jatuh cinta sama AI ini mah."

    elif label == "life":
        if w > 0.5:
            return "🌀 Alur hidupmu penuh plot twist, cocok jadi drama Netflix"
        elif confidence < 0.5:
            return "📈 Hidupmu naik turun, tapi kamu tetap senyum kayak nggak ada apa-apa"
        else:
            return "Kehidupanmu akan penuh perubahan yang menarik."

    return "Belum bisa membaca garis tersebut."

def generate_frames():
    global last_frame
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            print("⚠️ Frame tidak berhasil dibaca.")
            continue

        with frame_lock:
            last_frame = frame.copy()

        try:
            results = model(frame)[0]
            annotated = results.plot()
            ret, buffer = cv2.imencode('.jpg', annotated)
            if not ret:
                print("⚠️ Gagal meng-encode frame.")
                continue
            frame_bytes = buffer.tobytes()
        except Exception as e:
            print(f"❌ Error saat proses YOLO: {e}")
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    global last_frame
    with frame_lock:
        if last_frame is None:
            return jsonify({"error": "No frame available"}), 500
        frame = last_frame.copy()

    try:
        results = model(frame)[0]
        annotated = results.plot()

        ramalan_semua = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = results.names[cls_id]

            x, y, w, h = box.xywh[0].tolist()
            conf = box.conf[0].item()

            ramalan = interpret_lines(name, x, y, w, h, conf)
            ramalan_semua.append(f"[{name}] {ramalan}")

        _, buffer = cv2.imencode('.jpg', annotated)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": encoded_img,
            "ramalan": ramalan_semua
        })
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

# Cleanup saat aplikasi berhenti
import atexit
@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)
