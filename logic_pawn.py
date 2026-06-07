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

# 폰(Pawn) .obj 파일 로드
def load_obj(filename):
    vertices, faces = [], []
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
    pawn_vertices, pawn_faces = load_obj('pawn.obj')
    print(f"pawn.obj 로드 완료")
except Exception as e:
    print("pawn.obj 파일을 읽는 데 실패했습니다.")
    exit()

# 체스판 및 상태 변수 설정
tracker = HandTracker(confidence=0.7)
checkerboard = (7, 7)

objp_3d = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp_3d[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

rvec, tvec = None, None
is_locked = False
detection_start_time = None

# 폰 데이터에 팀(team), 진행 방향(dir), 첫 이동 여부(has_moved) 속성 추가
# 파란색 폰(Blue)은 Y=6에서 시작해 위로(-1) 전진, 빨간색 폰(Red)은 Y=1에서 시작해 아래로(+1) 전진
pawns = [
    {"id": 0, "pos": [3, 6], "color": (255, 0, 0), "alive": True, "team": "blue", "dir": -1, "has_moved": False},
    {"id": 1, "pos": [4, 1], "color": (0, 0, 255), "alive": True, "team": "red",  "dir": 1,  "has_moved": False}
]

holding_pawn_id = None 
current_held_pos = [0.0, 0.0] 
holding_pawn_start_pos = [0, 0] #무효한 이동일 때 되돌아갈 원래 위치 기억

cap = cv.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    is_pinch, index_pos = tracker.get_hand_state(frame)
    
    # 인식 단계
    if not is_locked:
        ret_chess, corners = cv.findChessboardCorners(gray, checkerboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK)
        if ret_chess:
            if detection_start_time is None: detection_start_time = time.time()
            elapsed_time = time.time() - detection_start_time
            remaining_time = 3.0 - elapsed_time

            if remaining_time <= 0:
                _, rvec, tvec = cv.solvePnP(objp_3d, corners, mtx, dist)
                is_locked = True
                print("3차원 체스판 공간이 고정되었습니다.")
            else:
                cv.putText(frame, f"STATUS: Found! Locking in {remaining_time:.1f}s...", (10, 30), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        else:
            detection_start_time = None
            cv.putText(frame, "STATUS: Searching Board...", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # [연산 및 렌더링 단계]
    if is_locked and rvec is not None and tvec is not None:
        grid_x, grid_y, finger_cell_x, finger_cell_y = -1, -1, -1, -1
        
        # Ray-Casting
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

        #드래그 앤 드롭, 규칙 검증
        if is_pinch and index_pos is not None:
            if holding_pawn_id is None:
                for pawn in pawns:
                    if pawn["alive"] and pawn["pos"] == [finger_cell_x, finger_cell_y]:
                        holding_pawn_id = pawn["id"]
                        # 기물을 집어 드는 순간 원래 위치를 백업해 둡니다.
                        holding_pawn_start_pos = pawn["pos"].copy()
                        break
            
            if holding_pawn_id is not None:
                cv.putText(frame, f"HOLDING PAWN {holding_pawn_id}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                current_held_pos = [grid_x, grid_y]
                target_cell_x, target_cell_y = finger_cell_x, finger_cell_y
                
        else: # 손을 놓았을 때 (Drop)
            if holding_pawn_id is not None:
                if 'target_cell_x' in locals() and 0 <= target_cell_x < 8 and 0 <= target_cell_y < 8:
                    
                    pawn = pawns[holding_pawn_id]
                    # 이동 거리와 방향 계산
                    dx = target_cell_x - holding_pawn_start_pos[0]
                    dy = target_cell_y - holding_pawn_start_pos[1]
                    direction = pawn["dir"]
                    
                    # 목표 칸에 다른 기물이 있는지 확인
                    target_pawn = None
                    for p in pawns:
                        if p["id"] != holding_pawn_id and p["alive"] and p["pos"] == [target_cell_x, target_cell_y]:
                            target_pawn = p
                            break
                            
                    is_valid_move = False
                    
                    # 체스 폰 규칙 1: 앞으로 1칸 이동 (앞이 비어있어야 함)
                    if dx == 0 and dy == direction and target_pawn is None:
                        is_valid_move = True
                        
                    # 체스 폰 규칙 2: 첫 이동 시 앞으로 2칸 (경로와 목표 칸 모두 비어있어야 함)
                    elif dx == 0 and dy == 2 * direction and not pawn["has_moved"]:
                        mid_y = holding_pawn_start_pos[1] + direction
                        path_clear = True
                        for p in pawns:
                            if p["alive"] and p["pos"] == [holding_pawn_start_pos[0], mid_y]:
                                path_clear = False
                        if path_clear and target_pawn is None:
                            is_valid_move = True
                            
                    # 체스 폰 규칙 3: 대각선 방향으로 1칸 공격 (반드시 적 기물이 있어야 함)
                    elif abs(dx) == 1 and dy == direction and target_pawn is not None:
                        if target_pawn["team"] != pawn["team"]:
                            is_valid_move = True

                    # 판별 결과 적용
                    if is_valid_move:
                        if target_pawn is not None:
                            target_pawn["alive"] = False
                            print(f"💥 {pawn['team'].upper()} 팀이 대각선 공격으로 폰 {target_pawn['id']}를 잡았습니다!")
                        pawn["pos"] = [target_cell_x, target_cell_y]
                        pawn["has_moved"] = True
                    else:
                        # 규칙에 어긋나면 원래 자리로 튕겨 나갑니다.
                        pawn["pos"] = holding_pawn_start_pos
                        print("폰은 그렇게 움직일 수 없습니다. 제자리로 돌아갑니다.")
                        
                else: # 체스판 밖으로 놓았을 때 제자리 복귀
                    pawns[holding_pawn_id]["pos"] = holding_pawn_start_pos
                    
                holding_pawn_id = None 

        # 렌더링
        for pawn in pawns:
            if not pawn["alive"]: continue 
            
            if holding_pawn_id == pawn["id"]:
                base_x, base_y = current_held_pos[0], current_held_pos[1]
                render_color, thickness = (0, 255, 255), 2 
            else:
                base_x, base_y = pawn["pos"][0] + 0.5, pawn["pos"][1] + 0.5
                render_color, thickness = pawn["color"], 1
                
            final_x, final_y, final_z = base_x - 0.7, base_y - 1.2, 0.0
            translated_vertices = pawn_vertices * OBJ_SCALE + np.array([final_x, final_y, final_z], dtype=np.float32)
            
            img_pts, _ = cv.projectPoints(translated_vertices, rvec, tvec, mtx, dist)
            img_pts = img_pts.astype(np.int32).reshape(-1, 2)
            
            for face in pawn_faces:
                pts = np.array([img_pts[idx] for idx in face], dtype=np.int32)
                cv.polylines(frame, [pts], isClosed=True, color=render_color, thickness=thickness)

    cv.imshow('AR Chess - Pawn Rules', frame)
    
    key = cv.waitKey(1)
    if key == 27: break
    elif key == ord('r') or key == ord('R'):
        is_locked = False
        rvec, tvec, detection_start_time = None, None, None
        for pawn in pawns: 
            pawn["alive"] = True
            pawn["has_moved"] = False
        pawns[0]["pos"], pawns[1]["pos"] = [3, 6], [4, 1] # 리셋 시 초기 위치로

cap.release()
cv.destroyAllWindows()
