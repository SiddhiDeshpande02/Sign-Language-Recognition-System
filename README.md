Sign Language Recognition
A real-time Sign Language Alphabet Recognition system that uses a webcam to recognize hand gestures representing the 26 English letters (A–Z). 
The project uses MediaPipe to extract hand landmarks and a Random Forest classifier to predict the corresponding alphabet.

Tech Stack
Python
OpenCV – Webcam and image processing
MediaPipe – Hand landmark detection
Scikit-learn – Random Forest classifier
NumPy – Numerical processing
Pickle – Dataset and model storage

How to Run
1. Install Dependencies
  pip install -r requirements.txt

2. Collect Images
  Run:
  python collect_imgs.py
  This collects 100 images for each of the 26 alphabet classes.

3. Create Dataset
  Run:
  python create_dataset.py
  This extracts hand landmarks from the collected images and creates data.pickle.

4. Train the Model
  Run:
  python train_classifier.py
  This trains the Random Forest classifier and saves the trained model as model.p.

5. Run Real-Time Recognition
  Run:
  python inference_classifier.py
  The webcam will open, and the system will detect the hand gesture and display the predicted alphabet in real time.
