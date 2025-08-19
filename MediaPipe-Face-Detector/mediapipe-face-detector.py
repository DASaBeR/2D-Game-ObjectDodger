import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection

def pick_main_face(boxes):
    if not boxes:
        return None
    areas = [(w*h) for (x, y, w, h) in boxes]
    return max(range(len(boxes)), key=lambda i: areas[i])

def draw_boxes(frame, boxes, main_idx):
    for i, (x, y, w, h) in enumerate(boxes):
        cx, cy = x + w // 2, y + h // 2
        color = (0, 255, 0) if i == main_idx else (255, 255, 0)
        thick = 3 if i == main_idx else 2
        label = f"{'MAIN ' if i == main_idx else ''}({cx}, {cy})"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)
        cv2.circle(frame, (cx, cy), 3, color, -1)
        cv2.putText(frame, label, (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        print(("MAIN " if i == main_idx else "Face ") + f"center: ({cx}, {cy})")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # model_selection: 0 ~ short range (2m), 1 ~ full range (5m+)
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        boxes = []
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                # Convert normalized bbox -> pixel bbox, clamp to frame
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = max(1, int(bb.width * w))
                bh = max(1, int(bb.height * h))
                # ensure within frame
                x = min(x, w - 1)
                y = min(y, h - 1)
                bw = min(bw, w - x)
                bh = min(bh, h - y)
                boxes.append((x, y, bw, bh))

        main_idx = pick_main_face(boxes)
        draw_boxes(frame, boxes, main_idx)

        cv2.imshow("MediaPipe Face Detection (Main = largest area)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
