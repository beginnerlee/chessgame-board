import cv2 as cv
import numpy as np
import time
from pinch import HandTracker

# 1. 카메라 캘리브레이션 데이터 로드
try:
    calib_data = np.load('calibration_data.npz')
    mtx = calib_data['mtx']
    dist = calib_data['dist']
    print("카메라 캘리브레이션 데이터를 성공적으로 불러왔습니다.")
except Exception as e:
    print("calibration_data.npz를 찾을 수 없어 기본값을 사용합니다.")
    mtx = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

# 2. 퀸(Queen) .obj 파일 로더
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
    queen_vertices, queen_faces = load_obj('queen.obj')
    print(f"queen.obj 로드 완료!")
except Exception as e:
    print("queen.obj 파일을 읽는 데 실패했습니다. 파일 경로를 확인하세요.")
    exit()

# 체스판 및 상태 변수 설정
tracker = HandTracker(confidence=0.7)
checkerboard = (7, 7)

objp_3d = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp_3d[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

rvec, tvec = None, None
is_locked = False
detection_start_time = None

# 퀸 데이터 (2개 배치)
queens = [
    {"id": 0, "pos": [0, 7], "color": (255, 0, 0), "alive": True, "team": "blue"},
    {"id": 1, "pos": [7, 0], "color": (0, 0, 255), "alive": True, "team": "red"}
]

holding_queen_id = None 
current_held_pos = [0.0, 0.0] 
holding_queen_start_pos = [0, 0]

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
                print("3차원 체스판 공간이 고정되었습니다!")
            else:
                cv.putText(frame, f"STATUS: Found! Locking in {remaining_time:.1f}s...", (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            detection_start_time = None
            cv.putText(frame, "STATUS: Searching Board...", (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(frame, "STATUS: 3D SPACE LOCKED", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    
    # [연산 및 렌더링 단계]
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

        # 퀸 이동 규칙 (드래그 앤 드롭)
        if is_pinch and index_pos is not None:
            if holding_queen_id is None:
                for q in queens:
                    if q["alive"] and q["pos"] == [finger_cell_x, finger_cell_y]:
                        holding_queen_id = q["id"]
                        holding_queen_start_pos = q["pos"].copy()
                        break
            
            if holding_queen_id is not None:
                cv.putText(frame, f"HOLDING QUEEN {holding_queen_id}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                current_held_pos = [grid_x, grid_y]
                
        else: # 손을 놓았을 때 (Drop)
            if holding_queen_id is not None:
                target_x, target_y = finger_cell_x, finger_cell_y
                
                if 0 <= target_x < 8 and 0 <= target_y < 8:
                    dx = abs(target_x - holding_queen_start_pos[0])
                    dy = abs(target_y - holding_queen_start_pos[1])
                    
                    # 퀸이동 규칙: 수평, 수직, 대각선 모두 가능
                    is_horizontal = (target_y == holding_queen_start_pos[1])
                    is_vertical = (target_x == holding_queen_start_pos[0])
                    is_diagonal = (dx == dy)
                    
                    if (is_horizontal or is_vertical or is_diagonal) and (dx != 0 or dy != 0):
                        target_queen = None
                        for q in queens:
                            if q["alive"] and q["pos"] == [target_x, target_y]:
                                target_queen = q
                                break
                        
                        if target_queen is None: # 이동 가능
                            queens[holding_queen_id]["pos"] = [target_x, target_y]
                        elif target_queen["team"] != queens[holding_queen_id]["team"]: # 캡처
                            target_queen["alive"] = False
                            queens[holding_queen_id]["pos"] = [target_x, target_y]
                            print(f"{queens[holding_queen_id]['team']} 퀸이 적을 잡았습니다!")
                        else:
                            print(" 아군 기물입니다.")
                    else:
                        print(" 퀸은 수직/수평/대각선으로만 이동 가능합니다.")
                
                holding_queen_id = None 

        # 렌더링
        offset_x, offset_y, offset_z = -0.7, -1.2, 0.0
        
        for q in queens:
            if not q["alive"]: continue 
            
            if holding_queen_id == q["id"]:
                base_x, base_y = current_held_pos[0], current_held_pos[1]
                render_color = (0, 255, 255)
                thickness = 2
            else:
                base_x, base_y = q["pos"][0] + 0.5, q["pos"][1] + 0.5
                render_color = q["color"]
                thickness = 1
                
            final_x = base_x + offset_x
            final_y = base_y + offset_y
            final_z = offset_z
            
            translated_vertices = queen_vertices * OBJ_SCALE + np.array([final_x, final_y, final_z], dtype=np.float32)
            img_pts, _ = cv.projectPoints(translated_vertices, rvec, tvec, mtx, dist)
            img_pts = img_pts.astype(np.int32).reshape(-1, 2)
            
            for face in queen_faces:
                pts = np.array([img_pts[idx] for idx in face], dtype=np.int32)
                cv.polylines(frame, [pts], isClosed=True, color=render_color, thickness=thickness)

    cv.imshow('AR Chess - Queen Rules', frame)
    
    key = cv.waitKey(1)
    if key == 27: break
    elif key == ord('r') or key == ord('R'):
        is_locked = False
        rvec, tvec, detection_start_time = None, None, None
        for q in queens: q["alive"] = True
        queens[0]["pos"], queens[1]["pos"] = [0, 7], [7, 0]

cap.release()
cv.destroyAllWindows()

