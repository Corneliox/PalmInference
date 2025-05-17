from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import threading
import atexit
import time

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

def get_euclidean_length(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def calculate_x_hand_ref(line_data):
    brain_line_coords = None
    feel_line_coords = None
    for line in line_data:
        name = line["name"].lower()
        if name in ["brain", "head"]:
            brain_line_coords = (line["x1"], line["x2"])
        elif name in ["feel", "heart"]:
            feel_line_coords = (line["x1"], line["x2"])

    if brain_line_coords and feel_line_coords:
        b_x1, b_x2 = brain_line_coords
        f_x1, f_x2 = feel_line_coords
        candidates = [
            abs(b_x1 - f_x1), abs(b_x1 - f_x2),
            abs(b_x2 - f_x1), abs(b_x2 - f_x2)
        ]
        x_ref = max(candidates) if candidates else 1.0
        return x_ref if x_ref > 0 else 1.0
    print("Warning: Essential lines for X_hand_ref missing. Using default.")
    return 200.0

# Thresholds for normalized values (relative to hand width)
THRESHOLDS = {
    "life": {
        "length": (0.33, 0.66),
        "height": (0.33, 0.66)
    },
    "heart": {
        "length": (0.33, 0.66),
        "height": (0.33, 0.66)
    },
    "head": {
        "length": (0.33, 0.66),
        "height": (0.33, 0.66)
    }
}

# Trait definitions per line
TRAITS = {
    "life": {
        "title": "🌱 Life Line",
        "length": {
            "high": {
                "result": "🔥 Enthusiastic (Antusias)",
                "explanation": "Anda menyambut hidup dengan semangat dan suka tantangan."
            },
            "mid": {
                "result": "🧘 Balanced Outlook (Pandangan Seimbang)",
                "explanation": "Anda menunjukkan antusiasme yang cukup namun tetap terkendali."
            },
            "low": {
                "result": "🌿 Cautious (Berhati-hati)",
                "explanation": "Anda cenderung mempertimbangkan semua langkah dengan teliti."
            }
        },
        "height": {
            "high": {
                "result": "⚡ Energetic (Berenergi)",
                "explanation": "Anda memiliki energi tinggi, aktif dan vital dalam kehidupan."
            },
            "mid": {
                "result": "🌗 Balanced Energy (Energi Seimbang)",
                "explanation": "Anda memiliki tingkat energi yang stabil antara aktif dan tenang."
            },
            "low": {
                "result": "🛡️ Conservative (Konservatif)",
                "explanation": "Anda lebih menyukai stabilitas dan cenderung berhati-hati."
            }
        }
    },
    "heart": {
        "title": "💓 Heart Line",
        "length": {
            "high": {
                "result": "💖 Expressive (Ekspresif)",
                "explanation": "Anda mudah menunjukkan perasaan dan membentuk hubungan emosional."
            },
            "mid": {
                "result": "💬 Emotionally Balanced (Seimbang secara Emosi)",
                "explanation": "Anda mampu mengontrol ekspresi perasaan secara tepat."
            },
            "low": {
                "result": "🨫 Introvert (Introvert)",
                "explanation": "Anda cenderung memproses perasaan secara pribadi dan tertutup."
            }
        },
        "height": {
            "high": {
                "result": "🌊 Emotional (Emosional)",
                "explanation": "Anda merasakan emosi dengan mendalam dan mudah empati."
            },
            "mid": {
                "result": "🌤️ Emotionally Balanced (Emosi Seimbang)",
                "explanation": "Anda tahu kapan harus merasa dan kapan harus berpikir logis."
            },
            "low": {
                "result": "🧠 Logical (Logis)",
                "explanation": "Anda lebih memilih berpikir rasional dibandingkan mengutamakan perasaan."
            }
        }
    },
    "head": {
        "title": "🧠 Head Line",
        "length": {
            "high": {
                "result": "🔍 Curious (Penasaran)",
                "explanation": "Anda suka mengeksplorasi ide baru dan memiliki rasa ingin tahu tinggi."
            },
            "mid": {
                "result": "📚 Balanced Thinking (Pikiran Seimbang)",
                "explanation": "Anda mampu menjaga keseimbangan antara keingintahuan dan fokus."
            },
            "low": {
                "result": "🎯 Focused (Fokus)",
                "explanation": "Anda langsung pada tujuan, praktis, dan tidak mudah terdistraksi."
            }
        },
        "height": {
            "high": {
                "result": "🎨 Creative (Kreatif)",
                "explanation": "Anda suka berpikir out-of-the-box dan penuh imajinasi."
            },
            "mid": {
                "result": "⚖️ Balanced Mindset (Pola Pikir Seimbang)",
                "explanation": "Anda bisa berpikir logis maupun kreatif tergantung situasi."
            },
            "low": {
                "result": "🧠 Logical (Logis)",
                "explanation": "Anda suka berpikir sistematis dan analitis dalam menyelesaikan masalah."
            }
        }
    }
}


def get_trait_by_value(value, low, high, trait_dict):
    if value > high:
        return trait_dict["high"]
    elif value > low:
        return trait_dict["mid"]
    else:
        return trait_dict["low"]


def interpret_traits(line_data):
    interpretations = []
    hand_width = calculate_x_hand_ref(line_data)

    for line in line_data:
        name = line["name"].lower()
        if name in ["feel", "heart"]:
            name = "heart"
        elif name in ["brain", "head"]:
            name = "head"
        elif name == "life":
            name = "life"
        else:
            continue  # skip unknown lines

        x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
        length = get_euclidean_length(x1, y1, x2, y2)
        height = abs(y2 - y1)

        normalized_length = length / hand_width
        normalized_height = height / hand_width

        low_len, high_len = THRESHOLDS[name]["length"]
        low_hgt, high_hgt = THRESHOLDS[name]["height"]

        traits = [
            get_trait_by_value(normalized_length, low_len, high_len, TRAITS[name]["length"]),
            get_trait_by_value(normalized_height, low_hgt, high_hgt, TRAITS[name]["height"])
        ]

        interpretations.append({
            "title": TRAITS[name]["title"],
            "traits": traits
        })

    return interpretations



# def generate_frames():
    # global last_frame, streaming
    # while True:
    #     if not streaming:
    #         black = np.zeros((480, 640, 3), dtype=np.uint8)
    #         ret, buffer = cv2.imencode('.jpg', black)
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    #         continue

    #     success, frame = cap.read()
    #     if not success:
    #         continue

    #     with frame_lock:
    #         last_frame = frame.copy()

    #     results = model(frame)[0]
    #     annotated = results.plot()
    #     ret, buffer = cv2.imencode('.jpg', annotated)
    #     if not ret:
    #         continue
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_frames():
    global last_frame, streaming
    last_detection_time = 0  
    detection_interval = 0  

    while True:
        if not streaming:
            black = np.zeros((720, 1280, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', black)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        success, frame = cap.read()
        if not success:
            continue

        current_time = time.time()

        with frame_lock:
            last_frame = frame.copy()

        
        if current_time - last_detection_time >= detection_interval:
            results = model(frame)[0]
            annotated = results.plot()
            last_detection_time = current_time
        else:
            annotated = frame 

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
