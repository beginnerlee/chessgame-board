import cv2 as cv
import numpy as np
import time
from pinch import HandTracker

# 카메라 캘리브레이션 데이터 로드
try:
    calib_data = np.load('calibration_data.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    print("카메라 캘리브레이션 데이터를 성공적으로 불러왔습니다.")
except Exception as e:
    print("calibration_data.npz를 찾을 수 없어 기본값을 사용합니다.")
    mtx = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

# 비숍(Bishop) .obj 파일 로더
def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                face = [int(parts.split('/')[0]) - 1 for parts in line.split()[1:]]
                faces.append(face)
    return np.array(vertices, dtype=np.float32), faces

OBJ_SCALE = 0.1  
try:
    # 파일명을 bishop.obj로 변경했습니다.
    bishop_vertices, bishop_faces = load_obj('bishop.obj')
    print(f"bishop.obj 로드 완료 (정점 수: {len(bishop_vertices)}, 면 수: {len(bishop_faces)})")
except Exception as e:
    print("bishop.obj 파일을 읽는 데 실패했습니다. 파일 경로를 확인하세요.")
    exit()

# 3. 체스판 및 상태 변수 설정
tracker = HandTracker(confidence=0.7)
checkerboard = (7, 7)

objp_3d = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp_3d[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

rvec, tvec = None, None
is_locked = False
detection_start_time = None

# 비숍 데이터 설정 (팀과 초기 위치 배정)
bishops = [
    {"id": 0, "pos": [2, 7], "color": (255, 0, 0), "alive": True, "team": "blue"},   # 파란색 비숍 (OpenCV BGR)
    {"id": 1, "pos": [5, 0], "color": (0, 0, 255), "alive": True, "team": "red"}     # 빨간색 비숍
]

holding_piece_id = None 
current_held_pos = [0.0, 0.0] 
holding_piece_start_pos = [0, 0] # 무효한 이동일 때 되돌아갈 원래 위치 기억

cap = cv.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    is_pinch, index_pos = tracker.get_hand_state(frame)
    
    # 3초 대기 후 3D 공간 포즈(solvePnP) 고정
    if not is_locked:
        ret_chess, corners = cv.findChessboardCorners(gray, checkerboard, 
                                                    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK)
        if ret_chess:
            if detection_start_time is None:
                detection_start_time = time.time()
            
            elapsed_time = time.time() - detection_start_time
            remaining_time = 3.0 - elapsed_time
            
            if remaining_time <= 0:
                # 호모그래피 대신 solvePnP를 사용하여 카메라와 체스판 사이의 3차원 위치 관계를 구합니다.
                _, rvec, tvec = cv.solvePnP(objp_3d, corners, mtx, dist)
                is_locked = True
                print("3차원 체스판 공간이 고정되었습니다")
            else:
                cv.putText(frame, f"STATUS: Found! Locking in {remaining_time:.1f}s...", (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            detection_start_time = None
            cv.putText(frame, "STATUS: Searching Board...", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(frame, "STATUS: 3D SPACE LOCKED", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # 연산 및 렌더링 단계
    if is_locked and rvec is not None and tvec is not None:
        
        # 1. Ray-Casting
        grid_x, grid_y = -1, -1
        finger_cell_x, finger_cell_y = -1, -1
        
        if index_pos is not None:
            pt = np.array([[[index_pos[0], index_pos[1]]]], dtype=np.float32)
            u, v = cv.undistortPoints(pt, mtx, dist)[0][0]
            P_cam = np.array([u, v, 1.0], dtype=np.float32)
            R_inv = cv.Rodrigues(rvec)[0].T
            A, B = R_inv @ P_cam, R_inv @ tvec.flatten()
            if A[2] != 0:
                s = B[2] / A[2]
                grid_x, grid_y = s * A[0] - B[0], s * A[1] - B[1]
                finger_cell_x, finger_cell_y = int(grid_x), int(grid_y)

        # 비숍 드래그 앤 드롭 및 대각선 규칙 검증
        if is_pinch and index_pos is not None:
            if holding_piece_id is None:
                for piece in bishops:
                    if piece["alive"] and piece["pos"] == [finger_cell_x, finger_cell_y]:
                        holding_piece_id = piece["id"]
                        holding_piece_start_pos = piece["pos"].copy()
                        break
            
            if holding_piece_id is not None:
                cv.putText(frame, f"HOLDING BISHOP {holding_piece_id}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                current_held_pos = [grid_x, grid_y]
                target_cell_x, target_cell_y = finger_cell_x, finger_cell_y
                
        else: # 손을 놓았을 때 (Drop)
            if holding_piece_id is not None:
                if 'target_cell_x' in locals() and 0 <= target_cell_x < 8 and 0 <= target_cell_y < 8:
                    
                    piece = bishops[holding_piece_id]
                    start_x, start_y = holding_piece_start_pos
                    dx = target_cell_x - start_x
                    dy = target_cell_y - start_y
                    
                    is_valid_move = False
                    
                    # 비숍 규칙 1: 완벽한 대각선 이동 판별 (|dx| == |dy|)
                    if abs(dx) == abs(dy) and dx != 0:
                        
                        # 이동 방향의 단위 벡터 (+1 또는 -1)
                        step_x = 1 if dx > 0 else -1
                        step_y = 1 if dy > 0 else -1
                        
                        curr_x, curr_y = start_x + step_x, start_y + step_y
                        path_clear = True
                        
                        # 비숍 규칙 2: 목적지에 도달하기 전 경로에 다른 기물이 있는지 대각선으로 스캔
                        while curr_x != target_cell_x and curr_y != target_cell_y:
                            for p in bishops:
                                if p["alive"] and p["pos"] == [curr_x, curr_y]:
                                    path_clear = False
                                    break
                            if not path_clear: break
                            curr_x += step_x
                            curr_y += step_y
                            
                        # 경로가 비어있다면 최종 목적지 확인
                        if path_clear:
                            target_piece = None
                            for p in bishops:
                                if p["id"] != holding_piece_id and p["alive"] and p["pos"] == [target_cell_x, target_cell_y]:
                                    target_piece = p
                                    break
                            
                            # 📜 비숍 규칙 3: 목적지가 비어있거나 적 기물일 때만 허용
                            if target_piece is None:
                                is_valid_move = True
                            elif target_piece["team"] != piece["team"]:
                                is_valid_move = True
                                target_piece["alive"] = False # 적 기물 캡처!
                                print(f"{piece['team'].upper()} 비숍이 날카로운 대각선 공격으로 적을 잡았습니다!")
                            else:
                                print("[규칙 위반] 목적지에 아군 기물이 있습니다.")
                        else:
                            print("[규칙 위반] 비숍은 다른 기물을 뛰어넘을 수 없습니다.")
                    else:
                        print("[규칙 위반] 비숍은 대각선으로만 이동해야 합니다.")

                    # 판별 결과 적용
                    if is_valid_move:
                        piece["pos"] = [target_cell_x, target_cell_y]
                    else:
                        piece["pos"] = holding_piece_start_pos # 무효 시 원래 위치로 튕겨냄
                        
                else: # 체스판 밖으로 놓았을 때 제자리 복귀
                    bishops[holding_piece_id]["pos"] = holding_piece_start_pos
                    
                holding_piece_id = None 

        # 렌더링 로직 (비숍 객체 사용)
        offset_x, offset_y, offset_z = -0.7, -1.2, 0.0  
        
        for piece in bishops:
            if not piece["alive"]: continue 
            
            if holding_piece_id == piece["id"]:
                base_x, base_y = current_held_pos[0], current_held_pos[1]
                render_color, thickness = (0, 255, 255), 2 
            else:
                base_x, base_y = piece["pos"][0] + 0.5, piece["pos"][1] + 0.5
                render_color, thickness = piece["color"], 1
                
            final_x, final_y, final_z = base_x + offset_x, base_y + offset_y, offset_z
            
            translated_vertices = bishop_vertices * OBJ_SCALE + np.array([final_x, final_y, final_z], dtype=np.float32)
            
            img_pts, _ = cv.projectPoints(translated_vertices, rvec, tvec, mtx, dist)
            img_pts = img_pts.astype(np.int32).reshape(-1, 2)
            
            for face in bishop_faces:
                pts = np.array([img_pts[idx] for idx in face], dtype=np.int32)
                cv.polylines(frame, [pts], isClosed=True, color=render_color, thickness=thickness)

    cv.imshow('AR Chess - Bishop Rules', frame)
    
    key = cv.waitKey(1)
    if key == 27: break
    elif key == ord('r') or key == ord('R'):
        is_locked = False
        rvec, tvec, detection_start_time = None, None, None
        for piece in bishops: piece["alive"] = True
        bishops[0]["pos"], bishops[1]["pos"] = [2, 7], [5, 0] # 리셋 시 초기 위치로

cap.release()
cv.destroyAllWindows()
