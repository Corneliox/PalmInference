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

# # def interpret_traits(line_data):
#     def get_length(x1, y1, x2, y2):
#         return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

#     interpretations = []

#     for line in line_data:
#         name = line["name"].lower()
#         x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
#         length = get_length(x1, y1, x2, y2)
#         height = abs(y2 - y1)

#         if name == "life":
#             if length > 180:
#                 interpretations.append({
#                     "title": "🌱 Life Line",
#                     "traits": [
#                         {
#                             "result": "✨ Energetic (Bersemangat)",
#                             "explanation": "Anda adalah seorang yang penuh semangat, berenergi tinggi, dan selalu antusias dalam menjalani hidup."
#                         }
#                     ]
#                 })
#             else:
#                 interpretations.append({
#                     "title": "🌱 Life Line",
#                     "traits": [
#                         {
#                             "result": "🛡️ Conservative (Konservatif)",
#                             "explanation": "Anda berhati-hati dalam mengambil keputusan dan cenderung menyukai stabilitas."
#                         }
#                     ]
#                 })

#         elif name in ["feel", "heart"]:
#             traits = []
#             if length > 170:
#                 traits.append({
#                     "result": "💖 Expressive (Ekspresif)",
#                     "explanation": "Anda menunjukkan perasaan dengan terbuka dan mudah membentuk hubungan emosional."
#                 })
#             else:
#                 traits.append({
#                     "result": "🨫 Introvert (Introvert)",
#                     "explanation": "Anda cenderung menyimpan perasaan dan lebih suka memprosesnya secara pribadi."
#                 })

#             if height > 60:
#                 traits.append({
#                     "result": "🌊 Emotional (Emosional)",
#                     "explanation": "Anda merasakan emosi dengan mendalam dan mudah terhubung secara empatik."
#                 })
#             else:
#                 traits.append({
#                     "result": "🧠 Logical (Logis)",
#                     "explanation": "Anda cenderung menggunakan logika dan berpikir rasional dalam hubungan."
#                 })

#             interpretations.append({
#                 "title": "💓 Heart Line",
#                 "traits": traits
#             })

#         elif name in ["brain", "head"]:
#             traits = []
#             if length > 160:
#                 traits.append({
#                     "result": "🔍 Curious (Penasaran)",
#                     "explanation": "Anda memiliki rasa ingin tahu yang besar dan suka mempelajari hal-hal baru."
#                 })
#             else:
#                 traits.append({
#                     "result": "🎯 Focused (Fokus)",
#                     "explanation": "Anda cenderung praktis dan langsung pada tujuan."
#                 })

#             if height > 50:
#                 traits.append({
#                     "result": "🎨 Creative (Kreatif)",
#                     "explanation": "Anda memiliki imajinasi yang kuat dan suka berpikir di luar kebiasaan."
#                 })
#             else:
#                 traits.append({
#                     "result": "🧠 Logical (Logis)",
#                     "explanation": "Anda lebih mengandalkan akal dan berpikir dengan struktur yang rapi."
#                 })

#             interpretations.append({
#                 "title": "🧠 Head Line",
#                 "traits": traits
#             })

#     return interpretations

def interpret_traits(line_data):
    def get_length(x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    interpretations = []

    for line in line_data:
        name = line["name"].lower()
        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        length = get_length(x1, y1, x2, y2)
        height = abs(y2 - y1)

        if name == "life":
            traits = []

            # Height-based trait (Energetic vs Conservative)
            if height > 100:
                traits.append({
                    "result": "⚡ Energetic (Berenergi)",
                    "explanation": "Anda memiliki energi fisik dan mental yang tinggi, cenderung aktif dan vital dalam kehidupan sehari-hari."
                })
            else:
                traits.append({
                    "result": "🛡️ Conservative (Konservatif)",
                    "explanation": "Anda lebih menyukai kestabilan, cenderung tidak impulsif, dan memilih pendekatan hati-hati."
                })

            # Length-based trait (Enthusiastic vs Cautious)
            if length > 180:
                traits.append({
                    "result": "🔥 Enthusiastic (Antusias)",
                    "explanation": "Anda memiliki antusiasme tinggi terhadap kehidupan dan menyambut tantangan dengan semangat."
                })
            else:
                traits.append({
                    "result": "🌿 Cautious (Berhati-hati)",
                    "explanation": "Anda cenderung menjaga diri dan mempertimbangkan langkah dengan teliti sebelum bertindak."
                })

            interpretations.append({
                "title": "🌱 Life Line",
                "traits": traits
            })

        elif name in ["feel", "heart"]:
            traits = []

            if length > 170:
                traits.append({
                    "result": "💖 Expressive (Ekspresif)",
                    "explanation": "Anda menunjukkan perasaan dengan terbuka dan mudah membentuk hubungan emosional."
                })
            else:
                traits.append({
                    "result": "🨫 Introvert (Introvert)",
                    "explanation": "Anda cenderung menyimpan perasaan dan lebih suka memprosesnya secara pribadi."
                })

            if height > 60:
                traits.append({
                    "result": "🌊 Emotional (Emosional)",
                    "explanation": "Anda merasakan emosi dengan mendalam dan mudah terhubung secara empatik."
                })
            else:
                traits.append({
                    "result": "🧠 Logical (Logis)",
                    "explanation": "Anda cenderung menggunakan logika dan berpikir rasional dalam hubungan."
                })

            interpretations.append({
                "title": "💓 Heart Line",
                "traits": traits
            })

        elif name in ["brain", "head"]:
            traits = []

            if length > 160:
                traits.append({
                    "result": "🔍 Curious (Penasaran)",
                    "explanation": "Anda memiliki rasa ingin tahu yang besar dan suka mempelajari hal-hal baru."
                })
            else:
                traits.append({
                    "result": "🎯 Focused (Fokus)",
                    "explanation": "Anda cenderung praktis dan langsung pada tujuan."
                })

            if height > 50:
                traits.append({
                    "result": "🎨 Creative (Kreatif)",
                    "explanation": "Anda memiliki imajinasi yang kuat dan suka berpikir di luar kebiasaan."
                })
            else:
                traits.append({
                    "result": "🧠 Logical (Logis)",
                    "explanation": "Anda lebih mengandalkan akal dan berpikir dengan struktur yang rapi."
                })

            interpretations.append({
                "title": "🧠 Head Line",
                "traits": traits
            })

    return interpretations

def generate_frames():
    global last_frame, streaming
    while True:
        if not streaming:
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
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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

        traits = interpret_traits(lines_info)

        annotated = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        trait_dict = {
            "life": next((t['traits'] for t in traits if "Life Line" in t['title']), None),
            "heart": next((t['traits'] for t in traits if "Heart Line" in t['title']), None),
            "head": next((t['traits'] for t in traits if "Head Line" in t['title']), None)
        }

        return jsonify({
            "image": encoded_img,
            "personality": trait_dict
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

@atexit.register
def cleanup():
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)