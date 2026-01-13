import cv2
import time
from ultralytics import YOLO

# 1. 설정값 (기존과 동일)
USER = "admin"
PASS = "miclab123"
IP = "192.168.0.9"

# 2. Orin 전용 GStreamer 파이프라인 구성 (핵심 수정 부분)
# rtspsrc: RTSP 수신 -> nvv4l2decoder: 하드웨어 디코딩 -> nvvidconv: 하드웨어 크기 조절
gst_pipeline = (
    f"rtspsrc location=rtsp://{USER}:{PASS}@{IP}:554/stream2 protocols=tcp latency=200 ! "
    "decodebin ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true sync=false max-buffers=1"
)

# 3. 모델 로드 (TensorRT .engine 사용 권장)
model = YOLO("yolov8n.engine", task="detect")

# 4. 비디오 캡처 (CAP_GSTREAMER 명시)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

# 디버깅 코드 추가
ret, frame = cap.read()
print("first read:", ret, None if frame is None else frame.shape)

if not cap.isOpened():
    print("GStreamer 파이프라인 확인 필요: RTSP 연결 실패")
    exit()

conf_thres = 0.4
process_every_n = 2  # GStreamer를 쓰면 성능이 좋아져서 2프레임당 1번도 충분히 가능합니다.
last_boxes = []
frame_idx = 0

while True:
    ret, frame = cap.read()  # GStreamer가 최신 프레임을 관리하므로 grab() 루프가 거의 필요없음
    if not ret:
        break

    frame_idx += 1

    # 추론 부분
    if frame_idx % process_every_n == 0:
        results = model.predict(frame, imgsz=416, conf=conf_thres, verbose=False, device=0)

        last_boxes = []
        r = results[0]
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()  # 정확도(확률) 가져오기
            clss = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                if int(cls) == 0:  # 사람(person)만 필터링
                    # 좌표와 정확도를 함께 저장
                    last_boxes.append((box.astype(int), conf))

    # 시각화 부분
    for (box, conf) in last_boxes:
        x1, y1, x2, y2 = box
        # 1. 사각형 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 2. 텍스트 추가 (예: person 85%)
        label = f"person {conf * 100:.1f}%"
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Orin Optimized RTSP", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()