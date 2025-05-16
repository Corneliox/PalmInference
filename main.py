import cv2
import gradio as gr
from ultralytics import YOLO

# Load your YOLO model
model = YOLO("palm.pt")  # Replace with your custom trained model

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

# Live inference function
def detect_and_display(frame):
    results = model(frame)[0]
    line_count = 0
    for box in results.boxes:
        cls_id = int(box.cls)
        name = results.names[cls_id]
        if name.lower() == "line":
            line_count += 1

    annotated_frame = results.plot()
    ramalan = interpret_lines(line_count)

    # Display text at the top
    y0 = 20
    for i, line in enumerate(ramalan.split("\n")):
        cv2.putText(annotated_frame, line, (10, y0 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return annotated_frame

# Launch Gradio interface
gr.Interface(fn=detect_and_display,
             inputs=gr.Image(source="webcam", streaming=True),
             outputs="image",
             live=True).launch()
