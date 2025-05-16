from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import threading
import atexit

app = Flask(__name__)
model = YOLO("palm.pt")

# Kamera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Kamera tidak dapat dibuka. Pastikan tidak digunakan oleh aplikasi lain.")

# Global variables
last_frame = None
frame_lock = threading.Lock()
streaming = False

#Ver 1
# Interpretasi garis tangan
# def interpret_lines(line_count):
    # if line_count >= 25:
    #     return "❤ Percintaanmu rumit seperti sinetron 300 episode.\n🌀 Alur hidupmu penuh plot twist, cocok jadi drama Netflix."
    # elif 15 <= line_count < 25:
    #     return "💘 Kamu tipe bucin yang suka bilang 'terserah' tapi ngambek diam-diam.\n📈 Hidupmu naik turun, tapi kamu tetap senyum kayak nggak ada apa-apa."
    # elif 8 <= line_count < 15:
    #     return "💞 Kamu jago PDKT, tapi suka kejebak zona teman.\n📚 Hidupmu seperti skripsi: ditunda-tunda tapi akhirnya kelar juga."
    # elif 3 <= line_count < 8:
    #     return "🪀 Percintaanmu jarang update, terakhir chat dari doi: 2 hari lalu.\n🚂 Alur hidupmu lurus kayak rel kereta... tapi kadang disabotase sinyal."
    # else:
    #     return "🔕 Cinta? Apa itu? Kayaknya kamu udah LDR sama jodoh sejak lahir.\n💤 Hidupmu tenang... terlalu tenang, kayak WiFi tetangga yang dikunci."

#Ver 2
def interpret_traits(line_data):
    def get_length(x1, y1, x2, y2):
        return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    interpretations = []

    for line in line_data:
        name = line["name"].lower()
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        length = get_length(x1, y1, x2, y2)
        height = abs(y2 - y1)

        if name == "life":
            if length > 180:
                interpretations.append({
                    "title": "Life Line",
                    "traits": [
                        ("Energetic", "Bersemangat", "Anda adalah seorang yang penuh semangat, berenergi tinggi, dan selalu antusias dalam menjalani hidup.")
                    ]
                })
            else:
                interpretations.append({
                    "title": "Life Line",
                    "traits": [
                        ("Conservative", "Konservatif", "Anda berhati-hati dalam mengambil keputusan dan cenderung menyukai stabilitas.")
                    ]
                })

        elif name == "feel" or name == "heart":
            traits = []
            if length > 170:
                traits.append(("Expressive", "Ekspresif", "Anda menunjukkan perasaan dengan terbuka dan mudah membentuk hubungan emosional."))
            else:
                traits.append(("Introvert", "Introvert", "Anda cenderung menyimpan perasaan dan lebih suka memprosesnya secara pribadi."))

            if height > 60:
                traits.append(("Emotional", "Emosional", "Anda merasakan emosi dengan mendalam dan mudah terhubung secara empatik."))
            else:
                traits.append(("Logical", "Logis", "Anda cenderung menggunakan logika dan berpikir rasional dalam hubungan."))
            interpretations.append({ "title": "Heart Line", "traits": traits })

        elif name == "brain" or name == "head":
            traits = []
            if length > 160:
                traits.append(("Curious", "Penasaran", "Anda memiliki rasa ingin tahu yang besar dan suka mempelajari hal-hal baru."))
            else:
                traits.append(("Focused", "Fokus", "Anda cenderung praktis dan langsung pada tujuan."))

            if height > 50:
                traits.append(("Creative", "Kreatif", "Anda memiliki imajinasi yang kuat dan suka berpikir di luar kebiasaan."))
            else:
                traits.append(("Logical", "Logis", "Anda lebih mengandalkan akal dan berpikir dengan struktur yang rapi."))
            interpretations.append({ "title": "Head Line", "traits": traits })

    # Format output
    output = ""
    for item in interpretations:
        output += f"{item['title']}\n"
        for trait in item['traits']:
            eng, indo, desc = trait
            output += f"- {eng} ({indo})\n-- {desc}\n"
        output += "\n"
    return output.strip()

# Generator stream
def generate_frames():
    global last_frame, streaming
    while True:
        if not streaming:
            # Pause: return a black frame or skip sending
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', black)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            continue

        with frame_lock:
            last_frame = frame.copy()

        results = model(frame)[0]
        annotated = results.plot()
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_stream():
    global streaming
    streaming = True
    return jsonify({"status": "stream started"})

@app.route('/stop', methods=['POST'])
def stop_stream():
    global streaming
    streaming = False
    return jsonify({"status": "stream stopped"})

# Ver 1
# @app.route('/capture', methods=['POST'])
# def capture():
    # global last_frame
    # with frame_lock:
    #     if last_frame is None:
    #         return jsonify({"error": "No frame available"}), 500
    #     frame = last_frame.copy()

    # try:
    #     results = model(frame)[0]
    #     line_count = 0
    #     for box in results.boxes:
    #         cls_id = int(box.cls)
    #         name = results.names[cls_id]
    #         if name.lower() == "line":
    #             line_count += 1

    #     annotated = results.plot()
    #     ramalan = interpret_lines(line_count)

    #     _, buffer = cv2.imencode('.jpg', annotated)
    #     encoded_img = base64.b64encode(buffer).decode('utf-8')

    #     return jsonify({
    #         "image": encoded_img,
    #         "ramalan": ramalan
    #     })
    # except Exception as e:
    #     return jsonify({"error": f"Processing failed: {e}"}), 500

#Ver 2
# @app.route('/capture', methods=['POST'])
# def capture():
    # global last_frame
    # with frame_lock:
    #     if last_frame is None:
    #         return jsonify({"error": "No frame available"}), 500
    #     frame = last_frame.copy()

    # try:
    #     results = model(frame)[0]
    #     line_count = 0
    #     lines_info = []  # To store line details: name and position

    #     for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
    #         name = results.names[int(cls_id)]
    #         if name.lower() in ["feel", "brain", "life"]:  # Or just check == "line" if that's your class
    #             line_count += 1
    #             x1, y1, x2, y2 = box.tolist()
    #             lines_info.append({
    #                 "name": name,
    #                 "x1": int(x1),
    #                 "y1": int(y1),
    #                 "x2": int(x2),
    #                 "y2": int(y2),
    #                 "center": {
    #                     "x": int((x1 + x2) / 2),
    #                     "y": int((y1 + y2) / 2)
    #                 }
    #             })

    #     annotated = results.plot()
    #     ramalan = interpret_lines(line_count)

    #     _, buffer = cv2.imencode('.jpg', annotated)
    #     encoded_img = base64.b64encode(buffer).decode('utf-8')

    #     return jsonify({
    #         "image": encoded_img,
    #         "ramalan": ramalan,
    #         "line_data": lines_info
    #     })

    # except Exception as e:
    #     return jsonify({"error": f"Processing failed: {e}"}), 500

#Ver 3
# @app.route('/capture', methods=['POST'])
# def capture():
    global last_frame
    with frame_lock:
        if last_frame is None:
            return jsonify({"error": "No frame available"}), 500
        frame = last_frame.copy()

    try:
        results = model(frame)[0]
        line_data = []

        for box in results.boxes:
            cls_id = int(box.cls)
            name = results.names[cls_id].lower()
            if name in ["feel", "life", "brain"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                line_data.append({
                    "name": name,
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "center": {"x": center_x, "y": center_y}
                })

        # Calculate hand span from feel and brain line only
        x_extremes = [d["x1"] for d in line_data if d["name"] in ["feel", "brain"]] + \
                     [d["x2"] for d in line_data if d["name"] in ["feel", "brain"]]
        if not x_extremes:
            return jsonify({"error": "No valid line detections"}), 500
        hand_span = max(x_extremes) - min(x_extremes)

        def calc_line_metrics(line):
            length = ((line["x2"] - line["x1"]) ** 2 + (line["y2"] - line["y1"]) ** 2) ** 0.5
            y_shape = abs(line["y2"] - line["y1"]) / hand_span
            norm_length = length / hand_span
            return norm_length, y_shape

        personality = {}
        for line in line_data:
            norm_len, y_shape = calc_line_metrics(line)
            if line["name"] == "life":
                personality["life"] = "🔥 Energetic" if norm_len > 0.5 else "🧩 Meticulous"
            elif line["name"] == "feel":
                personality["heart"] = (
                    "💖 Expressive, " if norm_len > 0.5 else "💭 Reserved, "
                ) + ("🧠 Logical" if y_shape < 0.3 else "💓 Emotional")
            elif line["name"] == "brain":
                personality["head"] = (
                    "📚 Curious, " if norm_len > 0.5 else "🎯 Practical, "
                ) + ("🖌 Creative" if y_shape > 0.3 else "🔍 Logical")

        # Annotated image
        annotated = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": encoded_img,
            "line_data": line_data,
            "personality": personality
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

#Ver 4
@app.route('/capture', methods=['POST'])
def capture():
    global last_frame
    with frame_lock:
        if last_frame is None:
            return jsonify({"error": "No frame available"}), 500
        frame = last_frame.copy()

    try:
        results = model(frame)[0]
        lines_info = []

        # Loop over detections
        for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
            name = results.names[int(cls_id)]
            if name.lower() in ["life", "heart", "feel", "brain", "head"]:
                x1, y1, x2, y2 = map(int, box.tolist())
                lines_info.append({
                    "name": name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

        personality = interpret_traits(lines_info)

        annotated = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "image": encoded_img,
            "personality": {
                "life": extract_section(personality, "Life Line"),
                "heart": extract_section(personality, "Heart Line"),
                "head": extract_section(personality, "Head Line")
            }
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

def extract_section(text, title):
    """Helper to extract each section from interpreted string output."""
    lines = text.splitlines()
    collecting = False
    section = []
    for line in lines:
        if line.strip() == title:
            collecting = True
            continue
        elif collecting and line.strip() and not line.startswith("-"):
            break
        elif collecting:
            section.append(line)
    return "\n".join(section).strip() if section else None


# Cleanup
@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)
