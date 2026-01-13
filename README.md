## 사용방법
---
### Yolo 모델 사용
터미널에서 
```
yolo export model=yolov8n.pt format=engine device=0 imgsz=416
```

이후 파일 실행

---
### testcode
Jetson 
H.264 테스트
gst-launch-1.0 -v rtspsrc location=rtsp://admin:miclab123@192.168.0.9:554/stream2 protocols=tcp latency=200 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! videoconvert ! autovideosink

H.265 테스트
gst-launch-1.0 -v rtspsrc location=rtsp://admin:miclab123@192.168.0.9:554/stream2 protocols=tcp latency=200 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! videoconvert ! autovideosink
