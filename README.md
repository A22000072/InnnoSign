
# TensorFlow Object Detection Pipeline

Proyek ini adalah implementasi pipeline deteksi objek menggunakan TensorFlow Object Detection API. Tujuan proyek ini adalah untuk mendeteksi objek dari gambar, video, atau live feed webcam dengan menggunakan model **SSD MobileNet V2**.

## 📋 Fitur
- **Model Pre-trained**: Menggunakan model SSD MobileNet V2 yang telah dilatih sebelumnya.
- **Fine-Tuning**: Melatih ulang model dengan data kustom.
- **Pipeline Lengkap**: Mulai dari persiapan dataset hingga evaluasi model.
- **Deteksi Real-Time**: Deteksi objek secara langsung menggunakan webcam.
- **Ekspor Model**: Mendukung ekspor model ke format TensorFlow Lite dan TensorFlow.js.

---

## 📂 Struktur Direktori
```
Tensorflow/
├── workspace/
│   ├── annotations/        # Folder untuk label map dan TFRecord
│   ├── images/             # Folder untuk dataset (train/test/live)
│   ├── models/             # Model hasil pelatihan
│   ├── pre-trained-models/ # Model yang diunduh dari TensorFlow
│   ├── scripts/            # Script tambahan (generate TFRecord)
│   ├── exported-models/    # Model yang sudah diekspor
```

---

## ⚙️ Instalasi dan Konfigurasi

### **1. Clone Repository**
Clone repository TensorFlow Models untuk mendapatkan Object Detection API:
```bash
git clone https://github.com/tensorflow/models Tensorflow/models
```

### **2. Install TensorFlow**
Pastikan TensorFlow versi yang sesuai terinstal:
```bash
pip install tensorflow==2.13.0
```

### **3. Unduh dan Pasang Protobuf Compiler**
Protobuf digunakan untuk mengkompilasi file `.proto`:
```bash
wget https://github.com/protocolbuffers/protobuf/releases/download/v27.2/protoc-27.2-linux-x86_64.zip
unzip protoc-27.2-linux-x86_64.zip -d Tensorflow/protoc
```

### **4. Kompilasi File Protobuf**
Navigasikan ke folder TensorFlow Models dan kompilasi file `.proto`:
```bash
cd Tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
```

### **5. Install TensorFlow Object Detection API**
Jalankan perintah berikut untuk menginstal API:
```bash
cd Tensorflow/models/research
pip install .
```

---

## 🏋️‍♂️ Pelatihan Model

### **1. Siapkan Dataset**
Dataset harus diletakkan di folder:
- `Tensorflow/workspace/images/train/`
- `Tensorflow/workspace/images/test/`

### **2. Buat TFRecord**
Gunakan script `generate_tfrecord.py` untuk membuat TFRecord:
```bash
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record
```

### **3. Konfigurasi Pipeline**
Update file `pipeline.config` dengan:
- **Path ke TFRecord dan label map**.
- **Jumlah kelas yang ingin dideteksi**.
- **Checkpoint model pre-trained**.

### **4. Jalankan Pelatihan**
Mulai pelatihan dengan:
```bash
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=2000
```

---

## 🔍 Evaluasi Model
Gunakan perintah berikut untuk mengevaluasi model:
```bash
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobnet
```

---

## 🚀 Ekspor Model
Ekspor model terlatih ke format SavedModel:
```bash
python Tensorflow/models/research/object_detection/exporter_main_v2.py --input_type=image_tensor --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --trained_checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobnet --output_directory=Tensorflow/workspace/models/my_ssd_mobnet/export
```

---

## 📊 Real-Time Detection
Jalankan script berikut untuk mendeteksi objek secara langsung melalui webcam:
```python
import cv2
import numpy as np
from object_detection.utils import visualization_utils as viz_utils

# Jalankan webcam dan tampilkan deteksi
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=0.5)
    cv2.imshow('Real-Time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

---

## 📂 Ekspor ke TensorFlow Lite dan TensorFlow.js
- **TensorFlow Lite**:
   ```bash
   python Tensorflow/models/research/object_detection/export_tflite_graph_tf2.py --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --trained_checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobnet --output_directory=Tensorflow/workspace/models/my_ssd_mobnet/tflite
   ```

- **TensorFlow.js**:
   ```bash
   tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model Tensorflow/workspace/models/my_ssd_mobnet/export Tensorflow/workspace/models/my_ssd_mobnet/tfjs
   ```

---

## 📬 Kontak
Jika ada pertanyaan, hubungi saya melalui email atau lihat repository GitHub ini.
