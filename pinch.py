import cv2 as cv
import mediapipe as mp
import math

class HandTracker:
    def __init__(self, confidence=0.7):
        # MediaPipe Hands 모듈 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=confidence, 
            min_tracking_confidence=confidence
        )
        self.threshold = 35  # Pinch 인식 임계값 (픽셀 거리)

    def get_hand_state(self, frame):
        """
        영상 프레임을 입력받아 손을 추적하고, 시각화 가이드를 그린 뒤
        (is_pinch, index_pos)를 반환합니다.
        - is_pinch: 엄지/검지가 맞닿았으면 True, 아니면 False
        - index_pos: 검지 손가락 끝의 (x, y) 픽셀 좌표 (손이 없으면 None)
        """
        h, w, _ = frame.shape
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        is_pinch = False
        index_pos = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 기본 뼈대 시각화
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 엄지(4번)와 검지(8번) 랜드마크 추출
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                
                # 정규화 좌표 -> 실제 픽셀 좌표 변환
                thumb_x, thumb_y = int(thumb.x * w), int(thumb.y * h)
                index_x, index_y = int(index.x * w), int(index.y * h)
                index_pos = (index_x, index_y)
                
                # 유클리드 거리 계산
                pixel_dist = math.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
                
                # 제스처 판정 및 시각화 가이드라인
                if pixel_dist < self.threshold:
                    is_pinch = True
                    cv.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3) # 초록색 선
                    cv.circle(frame, (index_x, index_y), 8, (0, 255, 0), cv.FILLED)
                else:
                    is_pinch = False
                    cv.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255), 1) # 빨간색 선
                    cv.circle(frame, (index_x, index_y), 8, (255, 0, 0), cv.FILLED)
                    
        return is_pinch, index_pos
