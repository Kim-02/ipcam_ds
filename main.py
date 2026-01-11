import cv2
import time
from ultralytics import YOLO

USER = "admin"
PASS = "miclab123"
IP   = "192.168.0.9"
url  = f"rtsp://{USER}:{PASS}@{IP}:554/stream2"  # 지연 줄이려면 stream2 권장

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("RTSP 연결 실패")

# 지연 감소 시도(환경에 따라 무시될 수 있음)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

conf_thres = 0.4
imgsz = 416                 # 640 -> 416으로 낮추면 더 빨라짐
process_every_n = 3         # 3프레임당 1번만 추론
drop_grabs = 5              # 매 루프마다 5프레임 버리고 최신 프레임만
display_w = 960             # 화면 표시 폭

frame_idx = 0
last_boxes = []  # 최근 탐지 결과를 저장해서 추론 안 하는 프레임에도 표시

while True:
    # 버퍼에 쌓인 프레임 버리기(최신 프레임 유지)
    for _ in range(drop_grabs):
        cap.grab()

    ret, frame = cap.retrieve()
    if not ret:
        time.sleep(0.2)
        continue

    frame_idx += 1

    # 표시/추론용으로 프레임 자체를 줄임 (예: 960폭 기준)
    scale = display_w / frame.shape[1]
    small = cv2.resize(frame, (display_w, int(frame.shape[0] * scale)))

    if frame_idx % process_every_n == 0:
        results = model.predict(
            source=small,
            imgsz=imgsz,
            conf=conf_thres,
            verbose=False
        )

        r = results[0]
        last_boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), c, cls in zip(boxes, confs, clss):
                if int(cls) == 0:  # person
                    last_boxes.append((int(x1), int(y1), int(x2), int(y2), float(c)))

    # 최근 결과를 계속 그려서 “부드럽게”
    vis = small
    for x1, y1, x2, y2, c in last_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"person {c:.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("RTSP + YOLO (low-latency)", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
