import cv2 as cv
import numpy as np
import time
from pinch import HandTracker

#카메라 캘리브레이션 데이터 로드
try:
    calib_data = np.load('calibration_data.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    print("카메라 캘리브레이션 데이터를 성공적으로 불러왔습니다.")
except Exception as e:
    print("calibration_data.npz를 찾을 수 없어 기본값을 사용합니다.")
    mtx = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

# obj 파일 로더 함수
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
    knight_vertices, knight_faces = load_obj('knight.obj')
    print(f"knight.obj 로드 완료!")
except Exception as e:
    print("knight.obj 파일을 읽는 데 실패했습니다. 파일 경로를 확인하세요.")
    exit()

# 체스판 및 상태 변수 설정
tracker = HandTracker(confidence=0.7)
checkerboard = (7, 7)

objp_3d = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp_3d[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

rvec, tvec = None, None
is_locked = False
detection_start_time = None

# 나이트 데이터
pieces = [
    {"id": 0, "pos": [1, 7], "color": (255, 0, 0), "alive": True, "team": "blue"},
    {"id": 1, "pos": [6, 0], "color": (0, 0, 255), "alive": True, "team": "red"}
]

holding_piece_id = None 
current_held_pos = [0.0, 0.0]
holding_piece_start_pos = [0, 0]

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
        
        # Ray-Casting
        grid_x, grid_y = -1, -1
        finger_cell_x, finger_cell_y = -1, -1
        
        if index_pos is not None:
            pt = np.array([[[index_pos[0], index_pos[1]]]], dtype=np.float32)
            pt_undistorted = cv.undistortPoints(pt, mtx, dist)
            u, v = pt_undistorted[0][0]
            P_cam = np.array([u, v, 1.0], dtype=np.float32)
            
            R, _ = cv.Rodrigues(rvec)
            R_inv = R.T
            A = R_inv @ P_cam
            B = R_inv @ tvec.flatten()
            
            if A[2] != 0:
                s = B[2] / A[2]
                grid_x = s * A[0] - B[0]
                grid_y = s * A[1] - B[1]
                finger_cell_x, finger_cell_y = int(grid_x), int(grid_y)

        # 드래그 앤 드롭 및 나이트 규칙 적용
        if is_pinch and index_pos is not None:
            if holding_piece_id is None:
                for piece in pieces:
                    if piece["alive"] and piece["pos"] == [finger_cell_x, finger_cell_y]:
                        holding_piece_id = piece["id"]
                        holding_piece_start_pos = piece["pos"].copy()
                        break
            
            if holding_piece_id is not None:
                cv.putText(frame, f"HOLDING PIECE {holding_piece_id}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                current_held_pos = [grid_x, grid_y]
                
        else: # 손을 놓았을 때 (Drop)
            if holding_piece_id is not None:
                target_x, target_y = finger_cell_x, finger_cell_y
                
                if 0 <= target_x < 8 and 0 <= target_y < 8:
                    dx = abs(target_x - holding_piece_start_pos[0])
                    dy = abs(target_y - holding_piece_start_pos[1])
                    
                    # 나이트 이동 규칙: L자 체크
                    is_valid_move = (dx == 2 and dy == 1) or (dx == 1 and dy == 2)
                    
                    if is_valid_move:
                        # 목적지 타겟 분석 (아군인지 적군인지 확인)
                        target_piece = None
                        for p in pieces:
                            if p["alive"] and p["pos"] == [target_x, target_y]:
                                target_piece = p
                                break
                        
                        if target_piece is None: # 이동 가능 (빈 칸)
                            pieces[holding_piece_id]["pos"] = [target_x, target_y]
                        elif target_piece["team"] != pieces[holding_piece_id]["team"]: # 캡처 (적군)
                            target_piece["alive"] = False
                            pieces[holding_piece_id]["pos"] = [target_x, target_y]
                            print(f"{pieces[holding_piece_id]['team']} 나이트가 적을 잡았습니다")
                        else:
                            print("아군 기물이 있는 곳으로 이동할 수 없습니다.")
                    else:
                        print("나이트는 L자로만 이동할 수 있습니다.")
                
                holding_piece_id = None

        # 렌더링
        offset_x, offset_y, offset_z = -0.7, -1.2, 0.0
        
        for piece in pieces:
            if not piece["alive"]: continue
            
            if holding_piece_id == piece["id"]:
                base_x, base_y = current_held_pos[0], current_held_pos[1]
                render_color = (0, 255, 255)
                thickness = 2
            else:
                base_x, base_y = piece["pos"][0] + 0.5, piece["pos"][1] + 0.5
                render_color = piece["color"]
                thickness = 1
                
            final_x = base_x + offset_x
            final_y = base_y + offset_y
            final_z = offset_z
            
            translated_vertices = knight_vertices * OBJ_SCALE + np.array([final_x, final_y, final_z], dtype=np.float32)
            img_pts, _ = cv.projectPoints(translated_vertices, rvec, tvec, mtx, dist)
            img_pts = img_pts.astype(np.int32).reshape(-1, 2)
            
            for face in knight_faces:
                pts = np.array([img_pts[idx] for idx in face], dtype=np.int32)
                cv.polylines(frame, [pts], isClosed=True, color=render_color, thickness=thickness)

    cv.imshow('AR Chess - Knight Rules', frame)
    
    key = cv.waitKey(1)
    if key == 27: break
    elif key == ord('r') or key == ord('R'):
        is_locked = False
        rvec, tvec, detection_start_time = None, None, None
        for p in pieces: p["alive"] = True
        pieces[0]["pos"], pieces[1]["pos"] = [1, 7], [6, 0]

cap.release()
cv.destroyAllWindows()
