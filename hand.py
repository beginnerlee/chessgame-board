import cv2
import mediapipe as mp

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(1) # 로지텍 웹캠 사용

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # 미디어파이프는 RGB 이미지를 받음
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # BGR로 복구하여 OpenCV 출력
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 뼈대 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 검지 끝 좌표 (Landmark 8번)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.imshow('Hand Tracking Check', image)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()
