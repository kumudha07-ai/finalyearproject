# Import necessary libraries
import os  # For interacting with the operating system
import pickle  # For serializing and de-serializing Python objects
import mediapipe as mp  # For utilizing the MediaPipe library for hand tracking
import cv2  # For image processing

# Set up MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Use EXACT same settings as collect_imgs.py (video mode works better)
hands = mp_hands.Hands(
    static_image_mode=False,  # Same as collect_imgs.py
    max_num_hands=1,
    min_detection_confidence=0.5,  # Same as collect_imgs.py
    min_tracking_confidence=0.5   # Same as collect_imgs.py
)

# Define the data directory
DATA_DIR = 'data/'

# Initialize lists to store data and corresponding labels
data = []
labels = []

# Counters for statistics
total_images = 0
detected_images = 0
failed_images = 0

print("=" * 50)
print("Creating Dataset from Hand Images")
print("Press any key to skip image, ESC to quit")
print("=" * 50)

# Loop through the directories in the data directory
for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue
    
    print(f"\nProcessing Class {dir_}...")
    class_detected = 0
    class_failed = 0
    
    # Loop through image files in the current directory
    image_files = sorted([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    # WARM-UP: Process first few images multiple times to initialize tracking
    warmup_count = min(5, len(image_files))
    print(f"  Warming up detector with first {warmup_count} images...")
    for i in range(warmup_count):
        img = cv2.imread(os.path.join(dir_path, image_files[i]))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Process multiple times for warm-up
            for _ in range(3):
                hands.process(img_rgb)
    print("  Warm-up complete!")
    
    for idx, img_path in enumerate(image_files):
        total_images += 1
        full_path = os.path.join(dir_path, img_path)
        
        # Read the image
        img = cv2.imread(full_path)
        if img is None:
            print(f"  Warning: Could not read {img_path}")
            failed_images += 1
            class_failed += 1
            continue
        
        # Flip image (since we flipped during capture)
        # img = cv2.flip(img, 1)  # Try uncommenting if detection fails
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image - try multiple times if needed
        results = None
        for attempt in range(3):  # Try up to 3 times
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                break

        # Create display image
        display_img = img.copy()

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            # Initialize lists for storing x and y coordinates
            x_ = []
            y_ = []
            
            # Loop through each detected hand (take only the first one)
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on display image
                mp_drawing.draw_landmarks(
                    display_img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract and store the x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize the data and append it to the data_aux list
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Append the normalized data to the main data list and the label to the labels list
                data.append(data_aux)
                labels.append(dir_)
                detected_images += 1
                class_detected += 1
                
                # Show success status
                cv2.putText(display_img, f'DETECTED - Class {dir_}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break  # Only process first hand
        else:
            failed_images += 1
            class_failed += 1
            # Show fail status
            cv2.putText(display_img, f'NO HAND - Class {dir_}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show progress
        cv2.putText(display_img, f'Image {idx+1}/{len(image_files)}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_img, f'Detected: {class_detected} | Failed: {class_failed}', (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the image
        cv2.imshow('Processing Images', display_img)
        
        # Wait for key (1ms) - increase to see images longer
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            print("\nQuitting early...")
            break
    
    print(f"  Class {dir_}: {class_detected} detected, {class_failed} failed")
    
    # Check if ESC was pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

# Save the data and labels as a dictionary in a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print("\n" + "=" * 50)
print("Dataset Creation Complete!")
print("=" * 50)
print(f"Total images processed: {total_images}")
print(f"Hands detected: {detected_images}")
print(f"Detection failed: {failed_images}")
print(f"Detection rate: {detected_images/total_images*100:.1f}%" if total_images > 0 else "N/A")
print(f"\nSaved to: data.pickle")
print("=" * 50)
