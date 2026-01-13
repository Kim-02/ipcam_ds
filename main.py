import cv2
from ultralytics import YOLO

# 1. 설정값
USER = "admin"
PASS = "miclab123"
IP = "192.168.0.9"

# 2. 해결 2: uridecodebin 기반 (RTSP 수신+디페이+디코딩 자동)
# - dec. 로 decodebin의 "decoded (raw)" 출력 pad를 받음
# - queue로 버퍼링/링킹 안정화
gst_pipeline = (
    f"uridecodebin uri=rtsp://{USER}:{PASS}@{IP}:554/stream2 name=dec "
    "dec. ! queue ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

# 3. 모델 로드 (TensorRT .engine)
model = YOLO("yolov8n.engine", task="detect")

# 4. 비디오 캡처
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# 먼저 열림 확인
if not cap.isOpened():
    print("GStreamer 파이프라인 확인 필요: RTSP 연결 실패")
    raise SystemExit(1)

# 디버깅: 첫 프레임 확인
ret, frame = cap.read()
print("first read:", ret, None if frame is None else frame.shape)
if not ret:
    print("첫 프레임 수신 실패: RTSP/디코딩 협상 문제 가능")
    cap.release()
    raise SystemExit(1)

conf_thres = 0.4
process_every_n = 2
last_boxes = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # 추론
    if frame_idx % process_every_n == 0:
        results = model.predict(frame, imgsz=416, conf=conf_thres, verbose=False, device=0)

        last_boxes = []
        r = results[0]
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                if int(cls) == 0:  # person만
                    last_boxes.append((box.astype(int), conf))

    # 시각화
    for (box, conf) in last_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"person {conf * 100:.1f}%"
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Orin Optimized RTSP (uridecodebin)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
