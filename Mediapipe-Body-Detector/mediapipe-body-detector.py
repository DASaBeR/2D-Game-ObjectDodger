# full_body_bbox_preview.py
# ------------------------------------------------------------
# Live webcam preview that draws a full-body bounding box using
# MediaPipe Pose. It finds visible pose landmarks, builds a box,
# makes it square-ish, computes the center (cx, cy), and draws it.
#
# Controls:
#   - Press 'q' to quit.
#
# Requirements:
#   pip install opencv-python mediapipe
# ------------------------------------------------------------

import cv2
import mediapipe as mp
import time

# ------------- Config -------------
WEBCAM_INDEX = 0
MIN_VISIBILITY = 0.4        # ### IMPORTANT: landmarks with visibility below this are ignored
POSE_MIN_DET_CONF = 0.6     # ### IMPORTANT: initial detection confidence for Pose
POSE_MIN_TRK_CONF = 0.5     # tracking confidence for Pose
MODEL_COMPLEXITY = 1        # 0/1/2; higher = more accurate, slower
SHOW_LANDMARKS = False      # toggle to show pose landmarks for debugging
SMOOTH_ALPHA = 0.25         # ### IMPORTANT: EMA smoothing factor for center (reduce jitter)
DRAW_SQUARE = True          # set False to draw raw rectangular bbox instead of square

# ------------- Setup -------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera permissions or index.")

# Optional: set a friendlier resolution (comment out if not supported by your cam)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=MODEL_COMPLEXITY,
    enable_segmentation=False,
    min_detection_confidence=POSE_MIN_DET_CONF,
    min_tracking_confidence=POSE_MIN_TRK_CONF
)

# Smoothed center point (initialized lazily on first detection)
smoothed_cx, smoothed_cy = None, None

# FPS helper
t_last = time.time()
fps = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Sometimes webcams hiccup; skip this frame gracefully.
            continue

        # Flip for a selfie-view; comment out if you prefer mirrored coordinates
        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe Pose
        res = pose.process(rgb)

        cx, cy = None, None  # raw center for this frame

        # --- Build a bounding box from visible landmarks ---
        if res.pose_landmarks and res.pose_landmarks.landmark:
            xs, ys = [], []
            for lm in res.pose_landmarks.landmark:
                # ### IMPORTANT: Only include fairly confident, in-frame landmarks
                if (0.0 <= lm.visibility <= 1.0) and (lm.visibility > MIN_VISIBILITY):
                    xs.append(lm.x * w)
                    ys.append(lm.y * h)

            if xs and ys:
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                box_w = x_max - x_min
                box_h = y_max - y_min

                # Center of the raw rectangle bbox
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0

                # --- Make it square-ish (optional) ---
                if DRAW_SQUARE:
                    side = max(box_w, box_h)
                    x1 = cx - side / 2
                    y1 = cy - side / 2
                    x2 = cx + side / 2
                    y2 = cy + side / 2
                else:
                    x1, y1, x2, y2 = x_min, y_min, x_max, y_max

                # Clamp to frame boundaries
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                # --- Smoothing center with exponential moving average ---
                if smoothed_cx is None or smoothed_cy is None:
                    smoothed_cx, smoothed_cy = cx, cy
                else:
                    smoothed_cx = (1 - SMOOTH_ALPHA) * smoothed_cx + SMOOTH_ALPHA * cx
                    smoothed_cy = (1 - SMOOTH_ALPHA) * smoothed_cy + SMOOTH_ALPHA * cy

                # --- Draw bbox and center ---
                # ### IMPORTANT: This rectangle is the "full-body" bounding box you can reuse elsewhere
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # Draw raw center (small circle) and smoothed center (crosshair)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 255), -1)  # raw center
                scx, scy = int(smoothed_cx), int(smoothed_cy)
                cv2.drawMarker(frame, (scx, scy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                # Display coordinates (smoothed)
                cv2.putText(
                    frame,
                    f"Center: ({scx}, {scy})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        # Optionally draw landmarks/skeleton for debugging
        if SHOW_LANDMARKS and res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

        # FPS calc
        t_now = time.time()
        dt = t_now - t_last
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)  # simple smoothing
        t_last = t_now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Full-Body Bounding Box Preview (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pose.close()
    cap.release()
    cv2.destroyAllWindows()
