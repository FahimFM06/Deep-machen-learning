# 🎵 Music Genre Classification using Deep Learning

This project implements a **complete end-to-end Music Genre Classification system** using audio signal processing and deep learning.  
Multiple architectures are explored and compared, including:

- **CRNN (CNN + Bi-LSTM) – TensorFlow/Keras**
- **CRNN + BiGRU – PyTorch**
- **Spectrogram Transformer (ViT-style) – PyTorch**
- **Explainable AI (XAI)** using Integrated Gradients, Grad-CAM, and Occlusion

The project is built on the **GTZAN music genre dataset** and follows a clean, research-oriented pipeline.

---

## 📌 Project Highlights

- 🎧 Raw audio processing with **Librosa**
- 📊 Extensive **Exploratory Data Analysis (EDA)**
- 🎼 Log-Mel Spectrogram feature extraction
- 🧠 Multiple deep learning architectures (CNN, RNN, Transformer)
- 🔍 Explainable AI (XAI) for model interpretability
- ⚡ GPU-accelerated training (PyTorch & TensorFlow)
- 🌐 Streamlit-ready inference pipeline

---

## 🗂️ Dataset

- **Dataset:** GTZAN Music Genre Dataset  
- **Source:** https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification  
- **Genres (10):**  
- **blues, classical, country, disco, hiphop,jazz, metal, pop, reggae, rock.**
- - **Audio format:** WAV  
- **Duration:** ~30 seconds per track  
- **Sample rate:** 22,050 Hz  



---

## 🏗️ Project Pipeline

### 1️⃣ Audio Loading
- Loads WAV files genre-wise
- Preserves original sample rate
- Handles corrupted audio safely

### 2️⃣ Exploratory Data Analysis (EDA)
- Waveform visualization
- Genre distribution analysis
- Sample rate consistency check
- Track duration statistics
- Boxplots by genre
- Mel-spectrogram visualization

### 3️⃣ Audio Preprocessing
- Resample to **22,050 Hz**
- Convert to mono
- Fix length to **30 seconds**
- Split into **10 segments (3s each)**

### 4️⃣ Feature Extraction
- **Log-Mel Spectrogram**
- `n_mels = 128`
- `n_fft = 2048`
- `hop_length = 512`
- Per-segment normalization (zero mean, unit variance)
- Final input shape: **(128 × 130)**

---

## 🧠 Models Implemented

### 🔹 1. CRNN (TensorFlow / Keras)

**Architecture:**
- 3 × Conv2D + BatchNorm + MaxPooling
- Reshape CNN output → sequence
- **Bidirectional LSTM**
- Dense classification head

**Performance:**
- ✅ Test Accuracy: **~72%**

---

### 🔹 2. CRNN + BiGRU (PyTorch)

**Architecture:**
- CNN feature extractor
- Dynamic CNN output inference
- **Bidirectional GRU**
- Fully-connected classifier

**Performance:**
- ✅ Test Accuracy: **~79%**

This is the **best-performing model** in the project.

---

### 🔹 3. Spectrogram Transformer (PyTorch)

**Architecture:**
- Patch embedding of spectrograms
- Learnable `[CLS]` token
- Positional embeddings
- Transformer Encoder (Multi-Head Attention)
- Classification head

**Performance:**
- ✅ Test Accuracy: **~64%**

---

## 📊 Model Comparison

| Model | Framework | Test Accuracy |
|------|----------|---------------|
| CRNN (CNN + Bi-LSTM) | TensorFlow | ~72% |
| **CRNN + BiGRU** | **PyTorch** | **~79%** |
| Spectrogram Transformer | PyTorch | ~64% |

---

## 🔍 Explainable AI (XAI)

To understand **why** the model makes predictions, multiple XAI techniques are used:

### ✅ Integrated Gradients
- Highlights important time-frequency regions
- Shows contribution of spectrogram bins

### ✅ Grad-CAM
- Visualizes CNN attention regions
- Heatmap over spectrogram

### ✅ Occlusion Sensitivity
- Measures prediction sensitivity to masked regions

These methods improve **model transparency and trustworthiness**.

---

## 🖥️ Technologies Used

- **Python**
- **Librosa**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**
- **TensorFlow / Keras**
- **PyTorch**
- **Scikit-learn**
- **Captum (XAI)**
---

## 🚀 Streamlit Web Application (GUI)

This project includes an interactive **Streamlit-based web application** that allows users to upload audio files and receive real-time music genre predictions using the trained **CRNN + BiGRU (PyTorch)** model.

🔗 **Live App Link:**  
https://music-genre-classification-crnn-bigru.streamlit.app/

### 🎛️ GUI Features
- Audio upload with playback
- Device information (CPU / GPU)
- Input spectrogram size display
- Adjustable inference settings
- Top-K genre probability visualization
- Segment-wise prediction analysis
- Waveform and log-mel spectrogram visualization

---

## 🖥️ Streamlit App Layout

### 🔹 Sidebar Controls
- **Navigation**
  - Dashboard
  - How it works
  - About
- **Inference Settings**
  - Aggregation method (Mean / Median / Max)
  - Top-K predictions slider
  - Toggle:
    - Per-segment prediction table
    - Waveform visualization
    - Mel-spectrogram preview

### 🔹 Main Dashboard
- Displays:
  - Running device (CPU/GPU)
  - Input feature size (`128 × 130`)
  - Number of segments per track (`10`)
- Audio file uploader with supported formats:



---

## 🎛️ Audio Preprocessing Pipeline

The following preprocessing steps are applied before inference:

1. Resample audio to **22,050 Hz**
2. Convert to **mono**
3. Standardize audio length to **30 seconds**
4. Split audio into **10 segments**, each **3 seconds**
5. For each segment:
 - Compute **Log-Mel Spectrogram**
   - `n_mels = 128`
   - `n_fft = 2048`
   - `hop_length = 512`
 - Normalize features (zero mean, unit variance)
6. Run inference on each segment
7. Aggregate predictions to produce the final genre

**Final input shape per segment:**  

