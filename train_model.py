import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA

face_size = (128, 128)

def load_image(image_path):
    """
    Membaca gambar dari path dan mengonversinya menjadi grayscale.
    
    Args:
        image_path (str): Path file gambar.

    Returns:
        tuple: (gambar warna, gambar grayscale)
    """
    image = cv2.imread(image_path)
    if image is None:
        print('Error: Could not load image.')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30,30)):
    """
    Mendeteksi wajah pada gambar grayscale menggunakan Haar Cascade.

    Args:
        image_gray (numpy.ndarray): Gambar dalam format grayscale.
        scale_factor (float): Parameter untuk memperkecil gambar saat pencarian wajah.
        min_neighbors (int): Jumlah minimum neighbor untuk validasi wajah.
        min_size (tuple): Ukuran minimum wajah yang akan dideteksi.

    Returns:
        list: List bounding box wajah yang terdeteksi.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        image_gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors, 
        minSize=min_size
    )
    return faces

def crop_faces(image_gray, faces, return_all=False):
    """
    Memotong wajah dari gambar berdasarkan hasil deteksi wajah.

    Args:
        image_gray (numpy.ndarray): Gambar grayscale.
        faces (list): List koordinat wajah (x, y, w, h).
        return_all (bool): Jika True, potong semua wajah; jika False, hanya wajah terbesar.

    Returns:
        tuple: (list wajah terpotong, list bounding box wajah yang dipilih)
    """
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for x, y, w, h in faces:
                selected_faces.append((x, y, w, h))
                cropped_faces.append(image_gray[y:y+h, x:x+w])
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            selected_faces.append((x, y, w, h))
            cropped_faces.append(image_gray[y:y+h, x:x+w])
    return cropped_faces, selected_faces

def resize_and_flatten(face):
    """
    Resize wajah ke ukuran standar dan flatten menjadi 1D array.

    Args:
        face (numpy.ndarray): Gambar wajah.

    Returns:
        numpy.ndarray: Wajah yang telah di-flatten.
    """
    face_resized = cv2.resize(face, face_size)
    face_flattened = face_resized.flatten()
    return face_flattened

class MeanCentering(BaseEstimator, TransformerMixin):
    """
    Transformer custom untuk mean-centering data (mengurangi rata-rata).
    """
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self
    def transform(self, X):
        return X - self.mean_face

# Load dataset
dataset_dir = "images"
X, y = [], []

# Iterasi setiap gambar dalam folder dataset
for root, _, files in os.walk(dataset_dir):
    for f in files:
        path = os.path.join(root, f)
        _, gray = load_image(path)
        if gray is None:
            continue
        faces = detect_faces(gray)
        cropped, _ = crop_faces(gray, faces)
        if cropped:
            X.append(resize_and_flatten(cropped[0]))
            y.append(os.path.basename(root)) # Label diambil dari nama yang tertera di folder

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=177)

# Pipeline
pipe = Pipeline([
    ("center", MeanCentering()),
    ("pca", PCA(svd_solver="randomized", whiten=True, random_state=177)),
    ("svc", SVC(kernel="linear", random_state=177))
])

# Training pipeline
pipe.fit(X_train, y_train)

# Save model
with open("eigenface_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("✅ Training selesai dan model disimpan sebagai eigenface_pipeline.pkl")