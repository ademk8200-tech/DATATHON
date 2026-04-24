import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 1. Veriyi Oku
df = pd.read_csv("train.csv")

# EKSİK VERİLERİ TEMİZLE (Eğer yorum kısmı boşsa model çökmesin diye dolduruyoruz)
df['yorum'] = df['yorum'].fillna("bos_yorum")

# X (Girdi) ve y (Çıktı) ayarı
X = df['yorum']

# DİKKAT: 3. sütunun (hedef) adını aşağıdaki satıra tırnak içine yazın! (Örn: 'duygu', 'label', 'kategori' vb.)
y = df['HEDEF_SUTUNUN_ADINI_BURAYA_YAZIN']

# Veriyi Train/Test diye ikiye böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Kelimeleri Sayılara Çevir (TF-IDF Silahı)
# En çok geçen 5000 kelimeyi alarak makinenin kafasını karıştırmadan hızlı bir matris çıkarıyoruz
vectorizer = TfidfVectorizer(max_features=5000) 
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Modeli Eğit ve Skoru Gör
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Tahmin yap ve Skoru bas!
y_pred = model.predict(X_test_vec)
print("İLK NLP MODELİMİZİN F1 SKORU:", f1_score(y_test, y_pred, average='weighted'))