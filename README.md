# BooTani_ML - Forecasting Harga Pangan
## Deskripsi Proyek
Proyek ini merupakan bagian dari program Bangkit, bertujuan untuk melakukan forecasting harga pangan menggunakan teknik machine learning. Forecasting harga pangan adalah aspek penting dalam bidang ekonomi dan pertanian karena dapat membantu para petani, pedagang, dan pembuat kebijakan dalam mengambil keputusan yang lebih baik.

## Dataset
Dataset yang digunakan dalam proyek ini terdiri dari data historis harga berbagai komoditas pangan. Dataset ini berisi kolom-kolom berikut:
- Tahun
- Bulan
- Tanggal
- Nama Komoditas

## Model dan Implementasi
Untuk melakukan forecasting harga pangan, kami menggunakan model machine learning berikut:

### Implementasi
Implementasi model dilakukan dalam beberapa langkah:
1. **Preprocessing Data:** Melakukan pembersihan data, normalisasi, dan pembagian data menjadi set pelatihan dan pengujian.
2. **Pembangunan Model:** Membuat arsitektur LSTM menggunakan Keras dan TensorFlow.
3. **Pelatihan Model:** Melatih model menggunakan data pelatihan dan melakukan tuning hyperparameter untuk mendapatkan hasil terbaik.
4. **Evaluasi Model:** Mengevaluasi kinerja model menggunakan metrik seperti Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE).
5. **Konversi Model ke TensorFlow Lite:** Mengkonversi model yang telah dilatih ke format TensorFlow Lite untuk memudahkan deployment di perangkat dengan resource terbatas.
6. **Prediksi:** Menggunakan model yang telah dilatih dan dikonversi untuk memprediksi harga pangan di masa depan.

Berikut adalah potongan kode untuk pembangunan, pelatihan, dan konversi model LSTM ke TensorFlow Lite:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# Load dataset
data = pd.read_csv('dataset/harga_pangan.csv')
# Preprocessing data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['Harga'].values.reshape(-1, 1))

# Prepare training data
X_train, y_train = [], []
for i in range(60, len(data_scaled)):
    X_train.append(data_scaled[i-60:i, 0])
    y_train.append(data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Convert to TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Software dan Framework yang Digunakan
Proyek ini dibangun dengan menggunakan beberapa software dan framework utama sebagai berikut:
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow dan Keras
- TensorFlow Lite
- Matplotlib

## Cara Menggunakan
1. Clone repository ini.
2. Install semua dependensi yang dibutuhkan dengan menjalankan:
    ```bash
    pip install -r requirements.txt
    ```
3. Letakkan dataset di folder `dataset`.
4. Buka file `Forecasting_Harga_Pangan.ipynb` menggunakan Jupyter Notebook atau JupyterLab.
5. Jalankan semua sel dalam notebook untuk melatih model dan melakukan prediksi.
