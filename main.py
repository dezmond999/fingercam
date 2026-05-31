import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

last_x, last_y = None, None
canvas = None
cap = cv2.VideoCapture(0)

def is_pinching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.05

print("Управление:")
print("- Соедините большой и указательный пальцы = рисовать")
print("- Разъедините = прекратить рисование")
print("C - очистить | ESC - выход")

while True:
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    h, w, _ = frame.shape
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            
            pinching = is_pinching(hand_landmarks)
            color = (0, 255, 0) if pinching else (0, 0, 255)
            cv2.circle(frame, (x, y), 10, color, -1)
            
            if pinching:
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (x, y), (0, 255, 0), 5)
                last_x, last_y = x, y
            else:
                last_x, last_y = None, None
    
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("Рисование пальцами", combined)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('c') or key == ord('C'):
        canvas = np.zeros_like(frame)
        print("Холст очищен")

cap.release()
cv2.destroyAllWindows()
