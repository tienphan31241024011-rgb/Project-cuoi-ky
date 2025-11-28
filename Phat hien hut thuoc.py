from ultralytics import YOLO
import cv2
import winsound  

#Đường dẫn model
model_path = r"C:\Users\ACER\Downloads\best.pt"
model = YOLO(model_path)

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

# Cho phép thay đổi kích thước cửa sổ
cv2.namedWindow("Smoking Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smoking Detection", 1000, 800)

print("Đang khởi động nhận diện. Nhấn Q để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện
    results = model.predict(frame, conf=0.5)

    annotated_frame = results[0].plot()

    # CẢNH BÁO
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name.lower() in ["smoking", "smoke", "cigarette"]:
            print("Phát hiện người hút thuốc!")
            winsound.Beep(1000, 500)

    # Hiển thị video
    cv2.imshow("Smoking Detection", annotated_frame)

    # 👉 IN KÍCH THƯỚC CỬA SỔ
    h, w = annotated_frame.shape[:2]
    print(f"Kích thước cửa sổ: {w} x {h}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
