# main-game-ursina-mediapipe.py
from ursina import *
import cv2
import numpy as np
import os
import random
import time
import mediapipe as mp

class ObjectDodgerGame(Entity):
    def __init__(self):
        super().__init__()

        # Game state
        self.game_state = 'await_mode'     # 'await_mode' | 'running' | 'game_over'
        self.input_mode = None             # 'keyboard' | 'face'

        # Gameplay params
        self.player_speed = 0.1
        self.object_speed = 0.2
        self.max_speed = 3.0
        self.speed_increment = 0.005
        self.spawn_rate = 1.0  # seconds
        self.hearts = 3
        self.score = 0

        # Vision (created only if Face mode is chosen)
        self.cap = None
        self.mp_face = mp.solutions.face_detection
        self.face_detector = None
        self.face_confidence = 0.5
        self.face_model_selection = 1  # 0: short range (~2m), 1: full (~5m)

        # Smoothed target for face-driven motion
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
            position=(0, 0.2),
            scale=2,
            color=color.white,
            background=True,
            font='unifont-14.0.02.ttf'
        )

        # Only two buttons now: Keyboard & Face
        self.keyboard_button = Button(
            text="Keyboard (K)",
            color=color.orange,
            position=(-0.18, 0),
            scale=(0.35, 0.1),
            on_click=self.choose_keyboard
        )

        self.face_button = Button(
            text="Face Detection (F)",
            color=color.pink,
            position=(0.18, 0),
            scale=(0.35, 0.1),
            on_click=self.choose_face_detection
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
            text="Camera/Face detection unavailable. Please check your webcam.",
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

    def choose_keyboard(self):
        if self.game_state != 'await_mode':
            return
        self.input_mode = 'keyboard'
        self.hide_input_choice()
        # Start game immediately for keyboard
        self.start_game()

    def choose_face_detection(self):
        if self.game_state != 'await_mode':
            return
        self.input_mode = 'face'
        self.hide_input_choice()

        # Lazy init camera + MediaPipe, and only start game if fully OK
        self.camera_message.text = "Initializing camera…"
        self.camera_message.enabled = True
        success = self.init_face_pipeline()

        if success:
            self.camera_message.enabled = False
            self.camera_error.enabled = False
            self.start_game()
        else:
            # Stay in await_mode so nothing spawns; allow user to pick keyboard
            self.camera_message.enabled = False
            self.camera_error.enabled = True
            self.input_choice_text.text = "Choose Control Mode (camera failed)"
            self.input_choice_text.enabled = True
            self.keyboard_button.enabled = True
            self.face_button.enabled = True
            self.input_mode = None  # back to no mode

    # ---------- Face pipeline ----------
    def init_face_pipeline(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap or not self.cap.isOpened():
                return False

            # Warm-up read to ensure frames are available
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return False

            # Create MediaPipe detector
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=self.face_model_selection,
                min_detection_confidence=self.face_confidence
            )
            return True
        except Exception as e:
            print(f"Face pipeline init error: {e}")
            return False

    # ---------- Game lifecycle ----------
    def start_game(self):
        # Called only after mode chosen & (for face) camera initialized successfully
        self.game_state = 'running'
        # Kick off spawns only now
        invoke(self.spawn_object, delay=self.spawn_rate)

    def spawn_object(self):
        if self.game_state != 'running':
            return  # Never spawn before running or after game over

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
        # Only runs after face mode selected & camera initialized & game running
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
                # Pick largest face
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

                    mapped_x = (cx / w) * 4 - 2          # [-2, 2]
                    mapped_z = -((cy / h) * 4)           # [0..-4]

                    # Smooth
                    self.target_pos.x = (1 - self.smooth_alpha) * self.target_pos.x + self.smooth_alpha * mapped_x
                    self.target_pos.z = (1 - self.smooth_alpha) * self.target_pos.z + self.smooth_alpha * mapped_z

                    self.target_pos.x = max(-2, min(2, self.target_pos.x))
                    self.target_pos.z = max(-4, min(0, self.target_pos.z))

            # Apply to player
            self.player.x = self.target_pos.x
            self.player.z = self.target_pos.z

        except Exception as e:
            print(f"Face processing error: {e}")

    # ---------- Per-frame update ----------
    def update(self):
        if self.game_state != 'running':
            return

        if self.input_mode == 'face':
            self.process_face_input()

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
                            self.game_over = True
                            self.show_game_over()
                        continue
                if obstacle.z < -10:
                    self.score += 1
                    self.score_text.text = f'Score: {self.score}'
                    destroy(obstacle)
            except:
                continue

        if self.object_speed < self.max_speed:
            self.object_speed += self.speed_increment * time.dt

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
        except:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass


# ---------- App setup ----------
app = Ursina(borderless=False)
window.title = "3D Object Dodger - Keyboard / MediaPipe Face Detection"

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

def on_exit():
    game.cleanup()

app.exit_func = on_exit
app.run()

# Safety fallback
try:
    game.cleanup()
except:
    pass
