# Import necessary libraries
import os  # For interacting with the operating system
import cv2  # For accessing the webcam and image processing
import mediapipe as mp  # For hand detection
import pickle  # For saving dataset

# Set up MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create hands detector (same settings as working test)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define the data directory
DATA_DIR = 'data'
PICKLE_FILE = 'data.pickle'

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and dataset size
number_of_classes = 6# Total number of classes for data collection
dataset_size = 1000  # Maximum number of images to capture per class

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Current class index
current_class = 0
counter = 0
is_capturing = False
hand_detected = False

# Dataset storage (landmark data)
all_data = []
all_labels = []

# Create directory for current class
def ensure_class_dir(class_idx):
    class_dir = os.path.join(DATA_DIR, str(class_idx))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    return class_dir

# Get current image count for a class
def get_image_count(class_idx):
    class_dir = os.path.join(DATA_DIR, str(class_idx))
    if os.path.exists(class_dir):
        return len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    return 0

# Reset (delete all images) for a class
def reset_class(class_idx):
    class_dir = os.path.join(DATA_DIR, str(class_idx))
    if os.path.exists(class_dir):
        for f in os.listdir(class_dir):
            if f.endswith('.jpg'):
                os.remove(os.path.join(class_dir, f))
    return 0

# Extract normalized landmark data
def extract_landmarks(hand_landmarks):
    x_ = []
    y_ = []
    data_aux = []
    
    # Get all coordinates
    for landmark in hand_landmarks.landmark:
        x_.append(landmark.x)
        y_.append(landmark.y)
    
    # Normalize coordinates
    for landmark in hand_landmarks.landmark:
        data_aux.append(landmark.x - min(x_))
        data_aux.append(landmark.y - min(y_))
    
    return data_aux

# Save dataset to pickle
def save_dataset():
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump({'data': all_data, 'labels': all_labels}, f)

# Display instructions
def draw_ui(frame, current_class, counter, is_capturing, hand_detected, total_samples):
    h, w = frame.shape[:2]
    
    # Draw semi-transparent overlay for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Title
    cv2.putText(frame, 'Sign Language Data Collection', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Hand detection status
    if hand_detected:
        cv2.putText(frame, 'HAND DETECTED', (250, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'NO HAND', (280, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Current status
    status = "CAPTURING" if is_capturing else "PAUSED"
    status_color = (0, 255, 0) if is_capturing else (0, 165, 255)
    cv2.putText(frame, f'Class: {current_class} | Images: {counter} | {status}', (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Dataset info
    cv2.putText(frame, f'Total Dataset Samples: {total_samples}', (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Controls
    cv2.putText(frame, 'Controls:', (20, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, 'S - Start/Stop | C - Capture Single', (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, 'N - Next Class | P - Previous Class', (20, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, 'R - Reset Class | ESC - Save & Quit', (20, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, '(Saves images + landmarks in real-time)', (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Progress bar
    progress = min(counter / dataset_size, 1.0)
    bar_width = 300
    cv2.rectangle(frame, (20, h - 40), (20 + bar_width, h - 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 40), (20 + int(bar_width * progress), h - 20), (0, 255, 0), -1)
    cv2.putText(frame, f'{counter}/{dataset_size}', (330, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# Ensure initial class directory exists
ensure_class_dir(current_class)
counter = get_image_count(current_class)

print("=" * 50)
print("Sign Language Data Collection")
print("(Saves images + landmarks in real-time!)")
print("=" * 50)
print("Controls:")
print("  S     - Start/Stop continuous capture")
print("  C     - Capture single image")
print("  N     - Next class")
print("  P     - Previous class")
print("  R     - Reset class (delete all images)")
print("  ESC   - Save dataset & Quit")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands
    results = hands.process(rgb_frame)
    hand_detected = results.multi_hand_landmarks is not None
    
    # Store current landmarks if detected
    current_landmarks = None
    if hand_detected:
        current_landmarks = extract_landmarks(results.multi_hand_landmarks[0])
    
    # Create display frame
    display_frame = frame.copy()
    
    # Draw hand landmarks if detected
    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Draw UI
    display_frame = draw_ui(display_frame, current_class, counter, is_capturing, hand_detected, len(all_data))
    
    # Show the frame
    cv2.imshow('Sign Language Data Collection', display_frame)
    
    # Continuous capture mode - ONLY SAVE IF HAND IS DETECTED
    if is_capturing and counter < dataset_size and hand_detected and current_landmarks:
        class_dir = ensure_class_dir(current_class)
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)  # Save original frame without landmarks
        
        # Save landmark data directly!
        all_data.append(current_landmarks)
        all_labels.append(str(current_class))
        
        counter += 1
        print(f'Captured: Class {current_class}, Image {counter}, Landmarks saved!')
    
    # Handle key presses
    key = cv2.waitKey(10) & 0xFF
    
    if key == 27:  # ESC - Save and Quit
        print("\nSaving dataset and exiting...")
        save_dataset()
        break
    
    elif key == ord('s') or key == ord('S'):  # S - Start/Stop capture
        is_capturing = not is_capturing
        if is_capturing:
            print(f"\n>>> Started capturing for Class {current_class} (waiting for hand...)")
        else:
            print(f"\n>>> Paused capturing. Class {current_class}: {counter} images")
    
    elif key == ord('c') or key == ord('C'):  # C - Capture single image
        if hand_detected and current_landmarks:
            if counter < dataset_size:
                class_dir = ensure_class_dir(current_class)
                img_path = os.path.join(class_dir, f'{counter}.jpg')
                cv2.imwrite(img_path, frame)
                
                # Save landmark data directly!
                all_data.append(current_landmarks)
                all_labels.append(str(current_class))
                
                counter += 1
                print(f'Captured: Class {current_class}, Image {counter}, Landmarks saved!')
            else:
                print(f"Class {current_class} already has {dataset_size} images!")
        else:
            print("Cannot capture - No hand detected!")
    
    elif key == ord('n') or key == ord('N'):  # N - Next class
        is_capturing = False
        current_class = (current_class + 1) % number_of_classes
        ensure_class_dir(current_class)
        counter = get_image_count(current_class)
        print(f"\n>>> Switched to Class {current_class} ({counter} existing images)")
    
    elif key == ord('p') or key == ord('P'):  # P - Previous class
        is_capturing = False
        current_class = (current_class - 1) % number_of_classes
        ensure_class_dir(current_class)
        counter = get_image_count(current_class)
        print(f"\n>>> Switched to Class {current_class} ({counter} existing images)")
    
    elif key == ord('r') or key == ord('R'):  # R - Reset class
        is_capturing = False
        counter = reset_class(current_class)
        # Also remove landmarks for this class from dataset
        all_data = [d for d, l in zip(all_data, all_labels) if l != str(current_class)]
        all_labels = [l for l in all_labels if l != str(current_class)]
        print(f"\n>>> Reset Class {current_class} - All images and landmarks deleted!")

# Release resources
hands.close()
cap.release()
cv2.destroyAllWindows()

# Final save
save_dataset()

print("\n" + "=" * 50)
print("Data collection complete!")
print("=" * 50)
print("Images Summary:")
for i in range(number_of_classes):
    count = get_image_count(i)
    print(f"  Class {i}: {count} images")
print(f"\nDataset saved to: {PICKLE_FILE}")
print(f"Total samples: {len(all_data)}")
print("=" * 50)
