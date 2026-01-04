import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
# Ganti 'nama_file.csv' dengan nama file dataset yang Anda unduh dari Kaggle
df = pd.read_csv('nama_file.csv')

# 2. Data Selection (Sesuai jurnal halaman 831 - Gambar 2) 
# Menghapus kolom identitas wilayah yang tidak dipakai prediksi
if 'Provinsi' in df.columns and 'Kab/Kota' in df.columns:
    df = df.drop(['Provinsi', 'Kab/Kota'], axis=1)

# 3. Data Cleaning (Sesuai jurnal halaman 831 - Gambar 3) [cite: 133]
# Menghapus baris yang targetnya (Klasifikasi Kemiskinan) kosong/NaN
df = df.dropna(subset=['Klasifikasi Kemiskinan'])

# 4. Transformation (Sesuai jurnal halaman 832 - Gambar 4) 
# Memastikan semua data bertipe numerik
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = pd.to_numeric(df[column], errors='coerce')

# Hapus sisa NaN jika ada setelah konversi
df = df.dropna()

# 5. Data Mining / Splitting (Sesuai jurnal halaman 832 - Gambar 5) [cite: 161, 167]
X = df.drop('Klasifikasi Kemiskinan', axis=1) # Fitur
y = df['Klasifikasi Kemiskinan']              # Target

# Split 80:20 seperti di jurnal
# Catatan: Jurnal menggunakan random_state tertentu tetapi terpotong di gambar.
# Kita gunakan 42 sebagai standar, atau coba angka lain jika hasil jauh berbeda.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Training Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 7. Evaluasi (Sesuai jurnal halaman 832 - Gambar 6) [cite: 173]
y_pred = dt_model.predict(X_test)

print("=== HASIL RE-TESTING ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Menampilkan Confusion Matrix (Sesuai jurnal halaman 833 - Gambar 8) [cite: 206]
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix Re-testing')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()