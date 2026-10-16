# Sign Language Classifier - Real-time Recognition
import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import os
import threading
from gtts import gTTS
from audioplayer import AudioPlayer

# Audio settings
last_spoken_time = 0
audio_lock = threading.Lock()

def speak_text(text):
    """Convert text to speech and play it"""
    global last_spoken_time
    
    current_time = time.time()
    # Don't repeat too quickly (wait 3 seconds between same messages)
    if current_time - last_spoken_time < 3:
        return
        
    last_spoken_time = current_time
    
    def _play_audio():
        with audio_lock:
            try:
                # Create audio file
                tts = gTTS(text=text, lang='en')
                temp_file = "temp_voice.mp3"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                tts.save(temp_file)
                
                # Play audio
                AudioPlayer(temp_file).play(block=True)
                
                # Clean up
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Audio Error: {e}")

    # Run in separate thread to not block video
    threading.Thread(target=_play_audio, daemon=True).start()

# Load the trained model
print("Loading model...")
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
print("Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Class labels mapping
labels_dict = {
    '0': 'How are you',
    '1': 'Hello',
    '2': 'I need water',
    '3': 'I need help',
    '4': 'I am hungry',
    '5': 'No',
    
}

# Settings
MARGIN = 20
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to display prediction
SMOOTHING_WINDOW = 5  # Number of frames to average for smoothing

# For prediction smoothing
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
confidence_history = deque(maxlen=SMOOTHING_WINDOW)

# FPS calculation
fps_start_time = time.time()
fps_counter = 0
fps = 0

# Statistics
total_predictions = 0
class_counts = {label: 0 for label in labels_dict.values()}

def get_smoothed_prediction(predictions, confidences):
    """Get the most common prediction from recent history"""
    if not predictions:
        return None, 0
    
    # Count occurrences
    pred_counts = {}
    for pred in predictions:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    # Get most common
    most_common = max(pred_counts, key=pred_counts.get)
    avg_confidence = np.mean([c for p, c in zip(predictions, confidences) if p == most_common])
    
    return most_common, avg_confidence

def draw_ui(frame, prediction, confidence, hand_detected, fps):
    """Draw the UI overlay"""
    h, w = frame.shape[:2]
    
    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, 'Convey Message', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # FPS
    cv2.putText(frame, f'FPS: {fps:.1f}', (w - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Status
    if hand_detected:
        status_text = "Hand Detected"
        status_color = (0, 255, 0)
    else:
        status_text = "Show your hand"
        status_color = (0, 165, 255)
    cv2.putText(frame, status_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Prediction display (bottom)
    if prediction and confidence > CONFIDENCE_THRESHOLD:
        # Large prediction box at bottom
        box_h = 80
        cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Prediction text
        cv2.putText(frame, prediction, (w//2 - 80, h - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        
        # Confidence bar
        bar_width = int((w - 40) * confidence)
        cv2.rectangle(frame, (20, h - box_h + 10), (w - 20, h - box_h + 25), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - box_h + 10), (20 + bar_width, h - box_h + 25), (0, 255, 0), -1)
        cv2.putText(frame, f'{confidence*100:.0f}%', (w - 60, h - box_h + 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Controls hint
    cv2.putText(frame, 'ESC to quit', (w - 100, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return frame

print("\n" + "=" * 50)
print("Sign Language Recognition - Real-time")
print("=" * 50)
print("Controls:")
print("  ESC - Quit")
print("=" * 50)
print("\nShow your hand to the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    results = hands.process(frame_rgb)
    
    hand_detected = results.multi_hand_landmarks is not None
    current_prediction = None
    current_confidence = 0
    
    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract features
            x_ = []
            y_ = []
            data_aux = []
            
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
            
            # Check feature count
            if len(data_aux) != 42:
                continue
            
            # Make prediction with probability
            try:
                prediction = model.predict([np.asarray(data_aux)])[0]
                probabilities = model.predict_proba([np.asarray(data_aux)])[0]
                confidence = max(probabilities)
                
                # Add to history for smoothing
                prediction_history.append(prediction)
                confidence_history.append(confidence)
                
                # Get smoothed prediction
                smoothed_pred, smoothed_conf = get_smoothed_prediction(
                    list(prediction_history), list(confidence_history)
                )
                
                if smoothed_pred is not None:
                    current_prediction = labels_dict.get(str(smoothed_pred), str(smoothed_pred))
                    current_confidence = smoothed_conf
                    
                    # Update statistics
                    if current_confidence > CONFIDENCE_THRESHOLD:
                        total_predictions += 1
                        if current_prediction in class_counts:
                            class_counts[current_prediction] += 1
                        
                        # Trigger audio
                        speak_text(current_prediction)
                
                # Draw bounding box
                x1 = int(min(x_) * W) - MARGIN
                y1 = int(min(y_) * H) - MARGIN
                x2 = int(max(x_) * W) + MARGIN
                y2 = int(max(y_) * H) + MARGIN
                
                # Color based on confidence
                if current_confidence > CONFIDENCE_THRESHOLD:
                    box_color = (0, 255, 0)  # Green
                else:
                    box_color = (0, 165, 255)  # Orange
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                
            except Exception as e:
                print(f"Prediction error: {e}")
    else:
        # Clear history when no hand detected
        prediction_history.clear()
        confidence_history.clear()
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()
    
    # Draw UI
    frame = draw_ui(frame, current_prediction, current_confidence, hand_detected, fps)
    
    # Show frame
    cv2.imshow('Sign Language Recognition', frame)
    
    # Handle key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

# Cleanup
hands.close()
cap.release()
cv2.destroyAllWindows()

# Print statistics
print("\n" + "=" * 50)
print("Session Statistics")
print("=" * 50)
print(f"Total predictions: {total_predictions}")
for label, count in class_counts.items():
    if total_predictions > 0:
        percentage = count / total_predictions * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
print("=" * 50)
