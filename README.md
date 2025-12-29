# PalmInference: Real-Time Palmistry Inference System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Inference%20Optimization-lightgrey)
![Status](https://img.shields.io/badge/Status-Active%20Optimization-orange)

**PalmInference** is a real-time computer vision application that utilizes a custom-trained **YOLOv8n** model to detect, segment, and analyze key hand palm lines (Life, Heart, and Head) from live webcam feeds. 

Designed for **CPU-only environments**, this project showcases the full Machine Learning lifecycle—from high-fidelity dataset curation to the deployment of optimized models using **ONNX Runtime**.

---

## 🚀 Key Features

* **Real-Time Inference:** High-speed detection pipeline optimized for consumer hardware using ONNX.
* **Cultural & Behavioral Analysis:** Translates visual features (line length, depth, curvature) into structured personality insights.
* **Hardware Optimization:** Leverages **ONNX Runtime** to bypass the need for heavy GPU compute, achieving significant speedups on standard CPUs.
* **Quantization Strategy:**
    * Implemented **Static Quantization** to minimize model footprint.
    * Actively managing a transition to **Dynamic Quantization** to recover precision (addressing a ~20% accuracy trade-off) while maintaining real-time latency.

---

## 🛠️ Technical Architecture

This project demonstrates advanced engineering trade-offs required for FinTech and high-performance AI solutions:

1.  **Data Curation:** Custom dataset annotation for precise palm line segmentation.
2.  **Model Training:** PyTorch-based training of YOLOv8 Nano architecture.
3.  **Deployment:** Export to `.onnx` format for platform-agnostic inference.
4.  **Version Control:** Rigorous **Git** workflow for feature branching and collaborative development.

---

## 📦 Installation

**Prerequisites:** Python 3.9+, Webcam.

```bash
# Clone the repository
git clone [https://github.com/Corneliox/PalmInference.git](https://github.com/Corneliox/PalmInference.git)

# Navigate to the directory
cd PalmInference

# Install dependencies (requires ultralytics, onnxruntime, opencv-python)
pip install -r requirements.txt
```

---

## 🖥️ Usage

To launch the inference engine with ONNX optimization enabled:

```bash
# Run on default webcam (Source 0)
python main.py --source 0 --model models/yolov8n-palm.onnx
```

To benchmark the quantized model performance:

```bash
python benchmark.py --quantized True
```

Quantization Still Under Developing

---

## 👥 Team & Collaboration

* **Raka Surya Kusuma** 
    * *Focus:* Architecture, Model Training (PyTorch).
* **Corneliox**
    * *Focus:* Frontend Inference Logic, Behavioral Analysis Mapping, Visualization, ONNX Optimization, Quantization Strategy.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.