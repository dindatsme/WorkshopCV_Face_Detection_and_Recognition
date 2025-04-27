# Face Recognition with PCA (Eigenfaces) and SVM

## 📄 Project Description

This project is part of a Computer Vision Workshop assignment. It implements a simple face recognition system using Principal Component Analysis (PCA), Eigenfaces, and Support Vector Machine (SVM) for classification.

The goal is to recognize faces in real-time through a webcam. You can also train the model with your own facial dataset.

---

## 📊 Repository Structure

```
.
├── images/             # Folder for training images (organized by person name)
├── result/             # Folder where you can see the example results (screenshots, recorded videos)
├── eigenface_pipeline.pkl  # Saved trained model 
├── train_model.py      # Script to train the model
├── realtime_webcam.py  # Script for real-time face recognition
├── Tutorial_CV_1_Meilany_517897.ipynb    # Jupyter notebook from the last tutorial
├── requirements.txt    # List of Python dependencies
├── README.md           # Program guide
```

---

## 🔧 How to Run Locally

1. **Clone or Download** this repository by run this code in your terminal.

    ```bash
    git clone https://github.com/dindatsme/WorkshopCV_Face_Detection_and_Recognition.git
    cd 'WorkshopCV_Face_Detection_and_Recognition'
    ```

2. **(Optional)** Create and activate a virtual environment.

    ```bash
    python -m venv .venv
    
    # Windows
    .venv\Scripts\activate
    
    # macOS/Linux
    source .venv/bin/activate
    ```

3. **Install Dependencies.**

    ```bash
    pip install -r requirements.txt
    ```

4. **(Optional)** Add your own face images.

    - Add your face images inside `images/your_name/`
    - Make sure each subfolder in `images/` is named after the person, containing their face pictures.

5. **Train the Model.**

    Run this if you added new face dataset
    
    ```bash
    python train_model.py
    ```
    
    This will generate or update `eigenface_pipeline.pkl`.

6. **Run Real-time Face Recognition.**

    ```bash
    python realtime_webcam.py
    ```
    
    Press `q` on the webcam window to exit.

7. **Check Results.**

    - The example of recorded video and screenshot of face recognition results can be seen inside the `result/` folder.

---

## 🔗 Notes

- If you have multiple webcams (internal + external), you may need to adjust the webcam index in `realtime_webcam.py`:
  
  ```python
  cap = cv2.VideoCapture(0)  # try 0, 1, or 2 depending on your webcam
  ```

- If you face a PowerShell execution policy error while activating the virtual environment, you can temporarily bypass it by running:

  ```bash
  Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  After everything have finished, you need to set it back by running:

  ```bash
  Set-ExecutionPolicy Restricted -Scope CurrentUser
  ```

- Make sure your face images are clear with various angles and lighting for better accuracy.

---
