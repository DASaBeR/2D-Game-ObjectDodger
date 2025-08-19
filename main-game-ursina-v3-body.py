# main-game-ursina-v3-body.py
from ursina import *
import cv2
import numpy as np
import os
import random
import time
import mediapipe as mp

"""
Object Dodger v3
- Adds a new control mode: 'Body'
- Uses MediaPipe Pose to compute a simple full-body bounding box (square-ish) from visible landmarks.
- Takes the center (cx, cy) of that bounding box and maps it to game-space just like the face-driven mode.
- Important parts are commented with "### IMPORTANT" for quick scanning.
"""

class ObjectDodgerGame(Entity):
    def __init__(self):
        super().__init__()

        # Game state
        self.game_state = 'await_mode'     # 'await_mode' | 'running' | 'game_over'
        self.input_mode = None             # 'keyboard' | 'face' | 'body'

        # Gameplay params
        self.player_speed = 0.1
        self.face_player_speed = 4.0
        self.body_player_speed = 4.0
        self.object_speed = 0.2
        self.max_speed = 3.0
        self.speed_increment = 0.005
        self.spawn_rate = 1.0  # seconds
        self.hearts = 3
        self.score = 0

        #FPS Helper
        self.last_time = time.time()
        self.fps = 0.0

        # Add an on-screen text for FPS
        self.fps_text = Text(
            text="FPS: 0",
            position=(0.7, 0.45),   # top-right corner
            scale=1.5,
            color=color.yellow,
            font='unifont-14.0.02.ttf'
        )

        # Vision (created lazily per mode)
        self.cap = None

        # Face detection (MediaPipe)
        self.mp_face = mp.solutions.face_detection
        self.face_detector = None
        self.face_confidence = 0.5
        self.face_model_selection = 1  # 0: short range (~2m), 1: full (~5m)

        # Body detection (MediaPipe Pose)
        self.mp_pose = mp.solutions.pose
        self.pose = None
        # Keep a slightly higher confidence since body landmarks are noisier
        self.pose_confidence = 0.6

        # Smoothed target for camera-driven motion (face/body)
        self.smooth_alpha = 0.25
        self.target_pos = Vec3(0, 0.5, -5)

        # Scene/UI
        self.create_scene()

        # Bind keyboard update (only acts if mode == keyboard & running)
        self.key_handler = Entity()
        self.key_handler.update = self.keyboard_input

    # ---------- Scene / UI ----------
    def create_scene(self):
        Sky(texture='sky_sunset')

        Entity(
            model='plane',
            texture='grass',
            scale=100,
            collider='mesh',
            position=(0, -1, 0),
            double_sided=True
        )

        self.player = Entity(
            model='sphere',
            color=color.green,
            scale=0.5,
            position=(0, 0.5, -5),
            collider='sphere'
        )

        self.obstacles = Entity()
        self.create_ui()

        DirectionalLight(parent=camera, position=(0, 10, -10), shadows=True)
        AmbientLight(color=color.rgba(100, 100, 100, 0.1))

    def create_ui(self):
        self.score_text = Text(
            text=f'Score: {self.score}',
            position=(-0.8, 0.45),
            scale=2,
            color=color.white,
            font='unifont-14.0.02.ttf'
        )

        self.hearts_text = Text(
            text=f'Lives: {self.hearts}',
            position=(-0.8, 0.4),
            scale=2,
            color=color.red,
            font='unifont-14.0.02.ttf'
        )

        self.input_choice_text = Text(
            text="Choose Control Mode",
            position=(0, 0.22),
            scale=2,
            color=color.white,
            background=True,
            font='unifont-14.0.02.ttf'
        )

        # Buttons: Keyboard, Face, Body
        self.keyboard_button = Button(
            text="Keyboard (K)",
            color=color.orange,
            position=(-0.33, 0),
            scale=(0.32, 0.1),
            on_click=self.choose_keyboard
        )

        self.face_button = Button(
            text="Face (F)",
            color=color.pink,
            position=(0.0, 0),
            scale=(0.32, 0.1),
            on_click=self.choose_face_detection
        )

        self.body_button = Button(
            text="Body (B)",
            color=color.azure,
            position=(0.33, 0),
            scale=(0.32, 0.1),
            on_click=self.choose_body_detection
        )

        self.camera_message = Text(
            text="Initializing camera…",
            position=(0, -0.2),
            scale=1.5,
            color=color.yellow,
            background=True,
            enabled=False,
            font='unifont-14.0.02.ttf'
        )

        self.camera_error = Text(
            text="Camera/Detection unavailable. Please check your webcam.",
            position=(0, -0.28),
            scale=1.3,
            color=color.red,
            background=True,
            enabled=False,
            font='unifont-14.0.02.ttf'
        )

        self.game_over_text = Text(
            text="",
            position=(0, 0.1),
            scale=3,
            color=color.red,
            background=True,
            enabled=False,
            font='unifont-14.0.02.ttf'
        )

        self.restart_button = Button(
            text="Play Again",
            color=color.green,
            position=(0, -0.1),
            scale=(0.3, 0.1),
            on_click=self.restart_game,
            enabled=False
        )

    # ---------- Mode selection ----------
    def hide_input_choice(self):
        self.input_choice_text.enabled = False
        self.keyboard_button.enabled = False
        self.face_button.enabled = False
        self.body_button.enabled = False

    def show_input_choice_with_error(self, msg="Choose Control Mode (camera failed)"):
        self.camera_message.enabled = False
        self.camera_error.text = msg
        self.camera_error.enabled = True
        self.input_choice_text.text = msg
        self.input_choice_text.enabled = True
        self.keyboard_button.enabled = True
        self.face_button.enabled = True
        self.body_button.enabled = True
        self.input_mode = None

    def choose_keyboard(self):
        if self.game_state != 'await_mode':
            return
        self.input_mode = 'keyboard'
        self.hide_input_choice()
        self.start_game()

    def choose_face_detection(self):
        if self.game_state != 'await_mode':
            return
        self.input_mode = 'face'
        self.hide_input_choice()
        self.camera_message.text = "Initializing camera…"
        self.camera_message.enabled = True
        success = self.init_camera() and self.init_face_pipeline()
        if success:
            self.camera_message.enabled = False
            self.camera_error.enabled = False
            self.start_game()
        else:
            self.show_input_choice_with_error()

    def choose_body_detection(self):
        if self.game_state != 'await_mode':
            return
        self.input_mode = 'body'
        self.hide_input_choice()
        self.camera_message.text = "Initializing camera…"
        self.camera_message.enabled = True
        success = self.init_camera() and self.init_body_pipeline()
        if success:
            self.camera_message.enabled = False
            self.camera_error.enabled = False
            self.start_game()
        else:
            self.show_input_choice_with_error()

    # ---------- Vision init / teardown ----------
    def init_camera(self):
        """Initialize webcam if needed."""
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                return False
            # Warm-up read to ensure frames are available
            ok, frame = self.cap.read()
            return bool(ok and frame is not None)
        except Exception as e:
            print(f"Camera init error: {e}")
            return False

    def init_face_pipeline(self):
        """Lazy-create MediaPipe FaceDetection."""
        try:
            if self.face_detector is None:
                self.face_detector = self.mp_face.FaceDetection(
                    model_selection=self.face_model_selection,
                    min_detection_confidence=self.face_confidence
                )
            return True
        except Exception as e:
            print(f"Face pipeline init error: {e}")
            return False

    def init_body_pipeline(self):
        """Lazy-create MediaPipe Pose."""
        try:
            if self.pose is None:
                # ### IMPORTANT: Pose in static_image_mode=False for video stream;
                # model_complexity=1 is a good balance; adjust if you want more accuracy.
                self.pose = self.mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=self.pose_confidence,
                    min_tracking_confidence=0.5
                )
            return True
        except Exception as e:
            print(f"Body pipeline init error: {e}")
            return False

    # ---------- Game lifecycle ----------
    def start_game(self):
        self.game_state = 'running'
        invoke(self.spawn_object, delay=self.spawn_rate)

    def spawn_object(self):
        if self.game_state != 'running':
            return

        obstacle = Entity(
            model='cube',
            color=color.red,
            scale=(0.5, 0.5, 0.5),
            position=(random.uniform(-2, 2), 0.25, 20),
            collider='box'
        )
        obstacle.parent = self.obstacles
        invoke(self.spawn_object, delay=self.spawn_rate)

    # ---------- Input handling ----------
    def keyboard_input(self):
        if self.input_mode != 'keyboard' or self.game_state != 'running':
            return

        if held_keys['left arrow'] or held_keys['a']:
            self.player.x -= self.player_speed
        if held_keys['right arrow'] or held_keys['d']:
            self.player.x += self.player_speed
        if held_keys['up arrow'] or held_keys['w']:
            self.player.z += self.player_speed
        if held_keys['down arrow'] or held_keys['s']:
            self.player.z -= self.player_speed

        self.player.x = max(-2, min(2, self.player.x))
        self.player.z = max(-4, min(0, self.player.z))

    def process_face_input(self):
        """Drive the player using the center of the largest detected face."""
        if self.input_mode != 'face' or self.game_state != 'running':
            return
        if self.cap is None or self.face_detector is None:
            return

        try:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.face_detector.process(rgb)

            if res.detections:
                boxes = []
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    x = max(0, int(bb.xmin * w))
                    y = max(0, int(bb.ymin * h))
                    bw = max(1, int(bb.width * w))
                    bh = max(1, int(bb.height * h))
                    x = min(x, w - 1)
                    y = min(y, h - 1)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    boxes.append((x, y, bw, bh))

                if boxes:
                    x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
                    cx = x + bw / 2
                    cy = y + bh / 2

                    # ### IMPORTANT: Map camera-space center -> game-space target
                    mapped_x = (cx / w) * 4 - 2          # [-2, 2]
                    mapped_z = -((cy / h) * 4)           # [0..-4]

                    # Smooth target to reduce jitter
                    self.target_pos.x = (1 - self.smooth_alpha) * self.target_pos.x + self.smooth_alpha * mapped_x
                    self.target_pos.z = (1 - self.smooth_alpha) * self.target_pos.z + self.smooth_alpha * mapped_z

                    self.target_pos.x = max(-2, min(2, self.target_pos.x))
                    self.target_pos.z = max(-4, min(0, self.target_pos.z))

            # Move player towards the target position at face_player_speed
            self._move_player_towards(self.target_pos, self.face_player_speed)

        except Exception as e:
            print(f"Face processing error: {e}")

    def process_body_input(self):
        """Drive the player using the center of a full-body bounding box computed from pose landmarks."""
        if self.input_mode != 'body' or self.game_state != 'running':
            return
        if self.cap is None or self.pose is None:
            return

        try:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)

            # ### IMPORTANT: Build a bounding box from visible landmarks
            if res.pose_landmarks and res.pose_landmarks.landmark:
                xs, ys = [], []
                for lm in res.pose_landmarks.landmark:
                    # Only include landmarks that are in-frame and likely tracked
                    if 0.0 <= lm.visibility <= 1.0 and lm.visibility > 0.4:
                        xs.append(lm.x * w)
                        ys.append(lm.y * h)

                if xs and ys:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    # Make it square-ish by expanding the shorter side (optional)
                    box_w = x_max - x_min
                    box_h = y_max - y_min
                    side = max(box_w, box_h)
                    cx = (x_min + x_max) / 2.0
                    cy = (y_min + y_max) / 2.0

                    # ### IMPORTANT: Map body center -> game-space (same mapping as face mode)
                    mapped_x = (cx / w) * 4 - 2          # [-2, 2]
                    mapped_z = -((cy / h) * 4)           # [0..-4]

                    # Smooth target to reduce jitter
                    self.target_pos.x = (1 - self.smooth_alpha) * self.target_pos.x + self.smooth_alpha * mapped_x
                    self.target_pos.z = (1 - self.smooth_alpha) * self.target_pos.z + self.smooth_alpha * mapped_z

                    self.target_pos.x = max(-2, min(2, self.target_pos.x))
                    self.target_pos.z = max(-4, min(0, self.target_pos.z))

            # Move player towards the target position at body_player_speed
            self._move_player_towards(self.target_pos, self.body_player_speed)

        except Exception as e:
            print(f"Body processing error: {e}")

    def _move_player_towards(self, target_vec3, max_step):
        """Helper to move the player toward a target with a clamped step size."""
        dx = (target_vec3.x * max_step) - self.player.x
        dz = (target_vec3.z * max_step) - self.player.z
        dist = (dx ** 2 + dz ** 2) ** 0.5
        if dist > 0:
            move_amount = min(max_step, dist)
            self.player.x += (dx / dist) * move_amount
            self.player.z += (dz / dist) * move_amount

        # Clamp player position
        self.player.x = max(-2, min(2, self.player.x))
        self.player.z = max(-4, min(0, self.player.z))

    # ---------- Per-frame update ----------
    def update(self):
        if self.game_state != 'running':
            return

        if self.input_mode == 'face':
            self.process_face_input()
        elif self.input_mode == 'body':
            self.process_body_input()

        # Move obstacles and detect collisions
        for obstacle in list(self.obstacles.children):
            if not obstacle.enabled:
                continue
            try:
                obstacle.z -= self.object_speed * time.dt * 60
                if -5 < obstacle.z < 5:
                    if obstacle.intersects(self.player).hit:
                        self.hearts -= 1
                        self.hearts_text.text = f'Lives: {self.hearts}'
                        destroy(obstacle)
                        if self.hearts <= 0:
                            self.show_game_over()
                        continue
                if obstacle.z < -10:
                    self.score += 1
                    self.score_text.text = f'Score: {self.score}'
                    destroy(obstacle)
            except Exception:
                continue

        if self.object_speed < self.max_speed:
            self.object_speed += self.speed_increment * time.dt
            
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            # Exponential smoothing for stability
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        self.fps_text.text = f"FPS: {self.fps:.1f}"

    # ---------- Game over / restart ----------
    def show_game_over(self):
        self.game_state = 'game_over'
        self.game_over_text.text = f"Game Over! Score: {self.score}"
        self.game_over_text.enabled = True
        self.restart_button.enabled = True
        self.save_high_score()

    def save_high_score(self):
        try:
            with open('highscores.txt', 'a') as f:
                f.write(f'{int(time.time())},{self.score}\n')
        except Exception as e:
            print(f"Error saving score: {e}")

    def restart_game(self):
        # Reset everything but stay in running with the same mode
        self.hearts = 3
        self.score = 0
        self.object_speed = 0.2
        self.score_text.text = f'Score: {self.score}'
        self.hearts_text.text = f'Lives: {self.hearts}'
        self.game_over_text.enabled = False
        self.restart_button.enabled = False

        for obstacle in self.obstacles.children:
            destroy(obstacle)

        self.player.position = (0, 0.5, -5)
        self.target_pos = Vec3(0, 0.5, -5)
        self.game_state = 'running'
        invoke(self.spawn_object, delay=self.spawn_rate)

    # ---------- Cleanup ----------
    def cleanup(self):
        try:
            if self.face_detector is not None:
                self.face_detector.close()
        except Exception:
            pass
        try:
            if self.pose is not None:
                self.pose.close()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass


# ---------- App setup ----------
app = Ursina(borderless=False)
window.title = "3D Object Dodger - Keyboard / Face / Body (MediaPipe)"

# Ensure font exists (safe to keep)
if not os.path.exists('unifont-14.0.02.ttf'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://unifoundry.com/pub/unifont/unifont-14.0.02/font-builds/unifont-14.0.02.ttf',
        'unifont-14.0.02.ttf'
    )

game = ObjectDodgerGame()
camera.position = (0, 10, -20)
camera.rotation_x = 30

# Quick key shortcuts for mode selection while on the menu
def input(key):
    if game.game_state == 'await_mode':
        if key.lower() == 'k':
            game.choose_keyboard()
        elif key.lower() == 'f':
            game.choose_face_detection()
        elif key.lower() == 'b':
            game.choose_body_detection()

def on_exit():
    game.cleanup()

app.exit_func = on_exit
app.run()

# Safety fallback
try:
    game.cleanup()
except Exception:
    pass
