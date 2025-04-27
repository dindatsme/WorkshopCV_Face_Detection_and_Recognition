import cv2
import pickle
import numpy as np
from train_model import MeanCentering, resize_and_flatten, detect_faces, crop_faces

# Load model
with open("eigenface_pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

def get_eigenface_score(X):
    """
    Menghitung confidence score dari hasil prediksi menggunakan decision function dari SVM.
    """
    X_pca = pipe[:2].transform(X) # Transformasi hingga PCA
    eigenface_scores = np.max(pipe[2].decision_function(X_pca), axis=1) # Ambil nilai skor prediksi tertinggi
    return eigenface_scores

def draw_text(image, label, score, pos=(0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_thickness=2, text_color=(0, 0, 0), text_color_bg=(0, 255, 0)):
    """
    Menulis label dan skor di atas bounding box wajah.
    """
    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y-h1-h2-25), (x + max(w1, w2)+20, y), text_color_bg, -1)
    cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

# Buka webcam
print("📷 Webcam starting...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Can't open camera!")
    exit()

# Loop utama
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)
    cropped_faces, selected_faces = crop_faces(gray, faces)
    if len(cropped_faces) > 0:
        X_face = []
        for face in cropped_faces:
            face_flattened = resize_and_flatten(face)
            X_face.append(face_flattened)
        
        X_face = np.array(X_face)
        labels = pipe.predict(X_face)
        scores = get_eigenface_score(X_face)

        # Loop semua wajah yang terdeteksi
        for (label, score, (x, y, w, h)) in zip(labels, scores, selected_faces):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_text(frame, label, float(score), (x, y)) 

    # Tampilkan hasil di layar
    cv2.imshow("Real-time Face Recognition", frame)
    
    # Tekan tombol "q" untuk exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()