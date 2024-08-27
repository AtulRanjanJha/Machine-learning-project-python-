import mediapipe as mp
import cv2
import numpy as np
import time

# Initialize MediaPipe Hands.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open webcam.
cap = cv2.VideoCapture(0)

# Function to detect gestures.
def detect_gestures(hand_landmarks):
    # Define landmarks for fingers.
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    # Calculate distances to detect if fingers are together or apart.
    dist_thumb_index = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    dist_index_middle = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([middle_tip.x, middle_tip.y]))
    dist_middle_ring = np.linalg.norm(np.array([middle_tip.x, middle_tip.y]) - np.array([ring_tip.x, ring_tip.y]))
    dist_ring_pinky = np.linalg.norm(np.array([ring_tip.x, ring_tip.y]) - np.array([pinky_tip.x, pinky_tip.y]))

    gestures = []

    # Detect numbers (1 to 5 fingers).
    if dist_thumb_index > 0.1 and dist_index_middle > 0.1 and dist_middle_ring > 0.1 and dist_ring_pinky > 0.1:
        if index_tip.y < thumb_tip.y and middle_tip.y > thumb_tip.y and ring_tip.y > thumb_tip.y and pinky_tip.y > thumb_tip.y:
            gestures.append(1)
        elif index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y > thumb_tip.y and pinky_tip.y > thumb_tip.y:
            gestures.append(2)
        elif index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y < thumb_tip.y and pinky_tip.y > thumb_tip.y:
            gestures.append(3)
        elif index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y:
            gestures.append(4)
        elif thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y:
            gestures.append(5)

    # Detect closed fist.
    if dist_thumb_index < 0.05 and dist_index_middle < 0.05 and dist_middle_ring < 0.05 and dist_ring_pinky < 0.05:
        gestures.append("fist")

    # Detect "equals" sign.
    if dist_index_middle < 0.05 and dist_middle_ring < 0.05 and dist_ring_pinky < 0.05 and dist_thumb_index > 0.1:
        gestures.append("equals")

    # Detect neutral (index and middle finger together).
    if dist_index_middle < 0.05 and dist_thumb_index > 0.1:
        gestures.append("neutral")

    # Detect Plus, Subtract, Multiply, and Divide based on hand gestures.
    if dist_thumb_index > 0.1 and dist_index_middle < 0.05 and dist_middle_ring > 0.1 and dist_ring_pinky > 0.1:
        gestures.append("plus")  # Example: Open hand with index finger up.
    elif dist_thumb_index < 0.05 and dist_index_middle > 0.1 and dist_middle_ring > 0.1 and dist_ring_pinky > 0.1:
        gestures.append("subtract")  # Example: Thumbs down gesture.
    elif dist_thumb_index > 0.1 and dist_index_middle < 0.05 and dist_middle_ring < 0.05 and dist_ring_pinky < 0.05:
        gestures.append("multiply")  # Example: Crossing index and middle fingers.
    elif dist_thumb_index > 0.1 and dist_index_middle < 0.05 and dist_middle_ring < 0.05 and dist_ring_pinky < 0.05:
        gestures.append("divide")  # Example: "V" shape with index and middle fingers.

    return gestures

# Function to evaluate the equation.
def evaluate_equation(equation):
    try:
        # Join the equation list to form a string.
        equation_str = ''.join(map(str, equation))
        # Evaluate the equation string.
        result = eval(equation_str.replace("=", ""))
        return result
    except:
        return "Error"

# Function to display the equation on the image.
def display_equation(image, equation):
    equation_str = ' '.join(map(str, equation))
    cv2.putText(image, equation_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# List to store the interpreted equation.
equation = []
# Track the previous gesture to avoid duplicates.
prev_gesture = None
# Timer for gesture debounce.
gesture_timer = 0
# Time interval between gestures in seconds.
gesture_interval = 1.0

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR to RGB.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip image horizontally for a later selfie-view display.
        image = cv2.flip(image, 1)
        
        # To improve performance, mark the image as not writeable.
        image.flags.writeable = False
        
        # Process the image and detect hands.
        results = hands.process(image)
        
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Interpret gestures and update the equation.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gestures = detect_gestures(hand_landmarks)
                
                # Process the detected gestures.
                for gesture in gestures:
                    current_time = time.time()
                    if gesture != prev_gesture and (current_time - gesture_timer > gesture_interval):
                        if isinstance(gesture, int):
                            equation.append(gesture)
                        elif gesture == "plus":
                            equation.append("+")
                        elif gesture == "subtract":
                            equation.append("-")
                        elif gesture == "multiply":
                            equation.append("*")
                        elif gesture == "divide":
                            equation.append("/")
                        elif gesture == "equals":
                            equation.append("=")
                            # Perform the calculation.
                            result = evaluate_equation(equation)
                            equation.append(result)
                        elif gesture == "reset":
                            equation.clear()
                        prev_gesture = gesture
                        gesture_timer = current_time
                    elif gesture == "neutral":
                        prev_gesture = None

                # Draw the hand landmarks.
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))
        
        # Display the interpreted gestures and equation.
        display_equation(image, equation)
        
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
