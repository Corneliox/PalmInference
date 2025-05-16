from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("palm.pt")
cap = cv2.VideoCapture(0)

def interpret_lines(line_count):
    if line_count >= 25:
        return "❤ Percintaanmu rumit seperti sinetron 300 episode.\n🌀 Alur hidupmu penuh plot twist, cocok jadi drama Netflix."
    elif 15 <= line_count < 25:
        return "💘 Kamu tipe bucin yang suka bilang 'terserah' tapi ngambek diam-diam.\n📈 Hidupmu naik turun, tapi kamu tetap senyum kayak nggak ada apa-apa."
    elif 8 <= line_count < 15:
        return "💞 Kamu jago PDKT, tapi suka kejebak zona teman.\n📚 Hidupmu seperti skripsi: ditunda-tunda tapi akhirnya kelar juga."
    elif 3 <= line_count < 8:
        return "🫀 Percintaanmu jarang update, terakhir chat dari doi: 2 hari lalu.\n🚂 Alur hidupmu lurus kayak rel kereta... tapi kadang disabotase sinyal."
    else:
        return "🔕 Cinta? Apa itu? Kayaknya kamu udah LDR sama jodoh sejak lahir.\n💤 Hidupmu tenang... terlalu tenang, kayak WiFi tetangga yang dikunci."

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        annotated = results.plot()
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500

    results = model(frame)[0]
    line_count = 0
    for box in results.boxes:
        cls_id = int(box.cls)
        name = results.names[cls_id]
        if name.lower() == "line":
            line_count += 1

    annotated = results.plot()
    ramalan = interpret_lines(line_count)

    _, buffer = cv2.imencode('.jpg', annotated)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": encoded_img,
        "ramalan": ramalan
    })

if __name__ == "__main__":
    app.run(debug=True)
