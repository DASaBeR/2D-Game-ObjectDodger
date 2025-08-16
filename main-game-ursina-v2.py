from ursina import *
import cv2
import numpy as np
import os
import random
import time

class ObjectDodgerGame(Entity):
    def __init__(self):
        super().__init__()
        
        # Game settings
        self.player_speed = 0.05
        self.object_speed = 0.5
        self.max_speed = 3.0
        self.speed_increment = 0.005
        self.spawn_rate = 1.0  # seconds
        self.hearts = 3
        self.score = 0
        self.game_over = False
        self.input_mode = None  # 'webcam' or 'keyboard' or 'face'
        self.webcam_available = False
        self.cap = None  # OpenCV video capture
        self.previous_frame = None
        self.face_cascade = None
        
        # Create game environment
        self.create_scene()
        
        # Setup webcam if available
        self.init_webcam()
        
        # Start game loop
        invoke(self.spawn_object, delay=self.spawn_rate)
        
        # Bind keys
        self.key_handler = Entity()
        self.key_handler.update = self.keyboard_input
        
    def create_scene(self):
        # Create sky
        Sky(texture='sky_sunset')
        
        # Create ground
        Entity(
            model='plane', 
            texture='grass', 
            scale=100, 
            collider='mesh',
            position=(0, -1, 0),
            double_sided=True
        )
        
        # Create player
        self.player = Entity(
            model='sphere', 
            color=color.green, 
            scale=0.5,
            position=(0, 0.5, -5),
            collider='sphere'
        )
        
        # Create obstacles container
        self.obstacles = Entity()
        
        # Create UI elements
        self.create_ui()
        
        # Create lighting
        DirectionalLight(parent=camera, position=(0, 10, -10), shadows=True)
        AmbientLight(color=color.rgba(100, 100, 100, 0.1))
    
    def create_ui(self):
        # Score display
        self.score_text = Text(
            text=f'Score: {self.score}',
            position=(-0.8, 0.45),
            scale=2,
            color=color.white,
            font='unifont-14.0.02.ttf'
        )
        
        # Hearts display
        self.hearts_text = Text(
            text=f'Lives: {self.hearts}',
            position=(-0.8, 0.4),
            scale=2,
            color=color.red,
            font='unifont-14.0.02.ttf'
        )
        
        # Input choice UI
        self.input_choice_text = Text(
            text="Choose Input Method:",
            position=(0, 0.2),
            scale=2,
            color=color.white,
            background=True,
            font='unifont-14.0.02.ttf'
        )
        
        self.webcam_button = Button(
            text="Webcam Motion (W)",
            color=color.blue,
            position=(-0.3, 0),
            scale=(0.3, 0.1),
            on_click=self.choose_webcam
        )
        
        self.keyboard_button = Button(
            text="Keyboard (K)",
            color=color.orange,
            position=(0, 0),
            scale=(0.3, 0.1),
            on_click=self.choose_keyboard
        )
        
        self.face_button = Button(
            text="Face Tracking (F)",
            color=color.pink,
            position=(0.3, 0),
            scale=(0.3, 0.1),
            on_click=self.choose_face_tracking
        )
        
        # Webcam connection message
        self.webcam_message = Text(
            text="Please connect your webcam...",
            position=(0, -0.2),
            scale=1.5,
            color=color.yellow,
            background=True,
            enabled=False,
            font='unifont-14.0.02.ttf'
        )
        
        # Game over screen
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
    
    def init_webcam(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.webcam_available = True
                # Load the face detection cascade
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                return True
            return False
        except Exception as e:
            print(f"Webcam initialization error: {e}")
            return False
    
    def choose_webcam(self):
        self.input_mode = 'webcam'
        self.hide_input_choice()
        
        if not self.webcam_available:
            self.webcam_message.enabled = True
            if not self.init_webcam():
                self.choose_keyboard()  # Fallback to keyboard if webcam fails
    
    def choose_keyboard(self):
        self.input_mode = 'keyboard'
        self.hide_input_choice()
    
    def choose_face_tracking(self):
        self.input_mode = 'face'
        self.hide_input_choice()
        
        if not self.webcam_available:
            self.webcam_message.enabled = True
            if not self.init_webcam():
                self.choose_keyboard()  # Fallback to keyboard if webcam fails
    
    def hide_input_choice(self):
        self.input_choice_text.enabled = False
        self.webcam_button.enabled = False
        self.keyboard_button.enabled = False
        self.face_button.enabled = False
    
    def process_webcam_input(self):
        if not self.webcam_available or self.cap is None:
            return
            
        try:
            # Get webcam frame
            ret, frame = self.cap.read()
            if not ret:
                return
                
            if self.input_mode == 'webcam':
                # Original motion detection logic
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if self.previous_frame is None:
                    self.previous_frame = gray
                    return
                    
                frame_delta = cv2.absdiff(self.previous_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) < 500:
                        continue
                    
                    (x, y, w, h) = cv2.boundingRect(contour)
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    self.player.x = (center_x / frame.shape[1]) * 4 - 2
                    self.player.z = (center_y / frame.shape[0]) * 2 - 3
                    
                    self.player.x = max(-2, min(2, self.player.x))
                    self.player.z = max(-4, min(0, self.player.z))
                
                self.previous_frame = gray
            
            elif self.input_mode == 'face':
                # Face detection logic
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Get the largest face (assuming it's the player)
                    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Calculate center of face
                    face_center_x = x + w / 2
                    face_center_y = y + h / 2
                    
                    # Map face position to game coordinates
                    # X position: map from 0-frame width to -2 to 2
                    self.player.x = (face_center_x / frame.shape[1]) * 4 - 2
                    # Z position: map from 0-frame height to -4 to 0 (reverse Y axis)
                    self.player.z = -((face_center_y / frame.shape[0]) * 4)
                    
                    # Constrain player movement
                    self.player.x = max(-2, min(2, self.player.x))
                    self.player.z = max(-4, min(0, self.player.z))
                    
        except Exception as e:
            print(f"Webcam processing error: {e}")
            self.webcam_available = False
            self.cap = None
    
    def keyboard_input(self):
        if self.input_mode != 'keyboard' or self.game_over:
            return
            
        # Arrow key movement
        if held_keys['left arrow'] or held_keys['a']:
            self.player.x -= self.player_speed
        if held_keys['right arrow'] or held_keys['d']:
            self.player.x += self.player_speed
        if held_keys['up arrow'] or held_keys['w']:
            self.player.z += self.player_speed
        if held_keys['down arrow'] or held_keys['s']:
            self.player.z -= self.player_speed
        
        # Constrain player movement
        self.player.x = max(-2, min(2, self.player.x))
        self.player.z = max(-4, min(0, self.player.z))
    
    def spawn_object(self):
        if self.game_over:
            return
            
        # Create new obstacle
        obstacle = Entity(
            model='cube',
            color=color.red,
            scale=(0.5, 0.5, 0.5),
            position=(random.uniform(-2, 2), 0.25, 20),
            collider='box'
        )
        
        # Add to obstacles container
        obstacle.parent = self.obstacles
        
        # Schedule next spawn
        invoke(self.spawn_object, delay=self.spawn_rate)
    
    def update(self):
        if self.game_over:
            return
            
        # Process input
        if self.input_mode in ['webcam', 'face']:
            self.process_webcam_input()
        
        # Move obstacles and check collisions
        for obstacle in self.obstacles.children.copy():
            if not obstacle.enabled:
                continue
                
            obstacle.z -= self.object_speed * time.dt * 60
            
            # Check collision
            if obstacle.z < -5:
                if obstacle.intersects(self.player).hit:
                    self.hearts -= 1
                    self.hearts_text.text = f'Lives: {self.hearts}'
                    destroy(obstacle)
                    
                    if self.hearts <= 0:
                        self.game_over = True
                        self.show_game_over()
                        return
            
            # Remove off-screen obstacles and increase score
            if obstacle.z < -10:
                self.score += 1
                self.score_text.text = f'Score: {self.score}'
                destroy(obstacle)
        
        # Increase difficulty
        if self.object_speed < self.max_speed:
            self.object_speed += self.speed_increment * time.dt
    
    def show_game_over(self):
        self.game_over_text.text = f"Game Over! Score: {self.score}"
        self.game_over_text.enabled = True
        self.restart_button.enabled = True
        
        # Save high score
        self.save_high_score()
    
    def save_high_score(self):
        try:
            with open('highscores.txt', 'a') as f:
                f.write(f'{time.time()},{self.score}\n')
        except Exception as e:
            print(f"Error saving score: {e}")
    
    def restart_game(self):
        # Reset game state
        self.hearts = 3
        self.score = 0
        self.object_speed = 0.5
        self.game_over = False
        
        # Reset UI
        self.score_text.text = f'Score: {self.score}'
        self.hearts_text.text = f'Lives: {self.hearts}'
        self.game_over_text.enabled = False
        self.restart_button.enabled = False
        
        # Clear obstacles
        for obstacle in self.obstacles.children:
            destroy(obstacle)
        
        # Reset player position
        self.player.position = (0, 0.5, -5)
        
        # Restart spawning
        invoke(self.spawn_object, delay=self.spawn_rate)

# Setup and run the game
app = Ursina(borderless=False)

# Set window title
window.title = "3D Object Dodger - Face Tracking"

# Download font that supports special characters if needed
if not os.path.exists('unifont-14.0.02.ttf'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://unifoundry.com/pub/unifont/unifont-14.0.02/font-builds/unifont-14.0.02.ttf',
        'unifont-14.0.02.ttf'
    )

# Start the game
game = ObjectDodgerGame()

# Camera setup
camera.position = (0, 10, -20)
camera.rotation_x = 30

app.run()

# Release webcam when done
if game.cap is not None:
    game.cap.release()