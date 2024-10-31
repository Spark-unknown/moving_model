# controllable model

import cv2
import mediapipe as mp
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize camera
cap = cv2.VideoCapture(0)

# 3D Object state
rotation_angle = [0, 0]  # x and y axis rotation angles
scale_factor = 1.0

def draw_cube():
    """Draws a 3D cube in OpenGL."""
    vertices = [
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def recognize_gesture(hand_landmarks):
    """Recognizes wrist movement for rotation and thumb-index spread gestures for scaling."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Calculate distances
    thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))
    
    # Determine scaling gestures
    if thumb_index_dist > 0.15:
        return "scale_up"
    elif thumb_index_dist < 0.05:
        return "scale_down"

    # Return rotation if wrist movement is detected
    return "rotate", wrist.x, wrist.y  # Using wrist position for rotation

# Initialize PyGame and OpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

previous_wrist_x = None
previous_wrist_y = None

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image for a mirrored display
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Gesture and manipulation logic
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        gesture = recognize_gesture(hand_landmarks)

        # Handle gestures
        if gesture == "scale_up":
            scale_factor += 0.01
        elif gesture == "scale_down":
            scale_factor = max(0.1, scale_factor - 0.01)
        elif isinstance(gesture, tuple) and gesture[0] == "rotate":
            wrist_x, wrist_y = gesture[1], gesture[2]
            if previous_wrist_x is not None and previous_wrist_y is not None:
                # Calculate change in wrist position
                dx = wrist_x - previous_wrist_x
                dy = wrist_y - previous_wrist_y

                # Adjust rotation angle based on wrist movement
                rotation_angle[0] += dy * 100  # Rotate around x-axis
                rotation_angle[1] += dx * 100  # Rotate around y-axis
            
            # Update previous wrist positions
            previous_wrist_x = wrist_x
            previous_wrist_y = wrist_y

        # Draw hand landmarks
        mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # OpenGL transformation for the 3D object
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPushMatrix()
    glScalef(scale_factor, scale_factor, scale_factor)
    glRotatef(rotation_angle[0], 1, 0, 0)
    glRotatef(rotation_angle[1], 0, 1, 0)
    draw_cube()
    glPopMatrix()
    pygame.display.flip()
    pygame.time.wait(10)
    
    # Display the image
    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()

# auto rotating model 
# import cv2
# import mediapipe as mp
# import numpy as np
# import pygame
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLU import *

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
# mp_drawing = mp.solutions.drawing_utils

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Initialize PyGame and OpenGL
# pygame.init()
# display = (800, 600)
# pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
# gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
# glTranslatef(0.0, 0.0, -5)

# # Model state
# model_pos = [0, 0, 0]
# is_grabbed = False
# rotation_angles = {'x': 0, 'y': 0, 'z': 0}  # Initial rotation angles
# rotation_speed = 0.1  # Initial rotation speed

# def draw_cube():
#     """Draws a colorful 3D cube in OpenGL."""
#     vertices = [
#         [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],  # Back face
#         [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1]     # Front face
#     ]
#     edges = [
#         (0, 1), (1, 2), (2, 3), (3, 0),  # Back face edges
#         (4, 5), (5, 6), (6, 7), (7, 4),  # Front face edges
#         (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
#     ]
#     colors = [
#         (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), 
#         (1, 0, 1), (0, 1, 1)  # Colors for the six sides
#     ]

#     # Draw cube faces
#     glBegin(GL_QUADS)
#     glColor3fv(colors[0])
#     for vertex in [0, 1, 2, 3]:
#         glVertex3fv(vertices[vertex])
#     glColor3fv(colors[1])
#     for vertex in [4, 5, 6, 7]:
#         glVertex3fv(vertices[vertex])
#     glColor3fv(colors[2])
#     glVertex3fv(vertices[3])
#     glVertex3fv(vertices[2])
#     glVertex3fv(vertices[6])
#     glVertex3fv(vertices[7])
#     glColor3fv(colors[3])
#     glVertex3fv(vertices[0])
#     glVertex3fv(vertices[1])
#     glVertex3fv(vertices[5])
#     glVertex3fv(vertices[4])
#     glColor3fv(colors[4])
#     glVertex3fv(vertices[3])
#     glVertex3fv(vertices[0])
#     glVertex3fv(vertices[4])
#     glVertex3fv(vertices[7])
#     glColor3fv(colors[5])
#     glVertex3fv(vertices[1])
#     glVertex3fv(vertices[2])
#     glVertex3fv(vertices[6])
#     glVertex3fv(vertices[5])
#     glEnd()

#     # Draw cube edges
#     glBegin(GL_LINES)
#     for edge in edges:
#         for vertex in edge:
#             glVertex3fv(vertices[vertex])
#     glEnd()

# def recognize_gesture(hand_landmarks):
#     """Recognizes gestures for controlling cube rotation speed and pause."""
#     global is_grabbed, rotation_speed
#     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#     index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

#     # Calculate distance for pinch gesture
#     thumb_index_dist = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y]))

#     # If pinch is detected, grab the cube
#     is_grabbed = thumb_index_dist < 0.05

#     # Detect thumb up/down for speed adjustment
#     if thumb_tip.y < wrist.y:  # Thumb up gesture
#         rotation_speed = min(rotation_speed + 0.01, 1)  # Limit max speed
#     elif thumb_tip.y > wrist.y:  # Thumb down gesture
#         rotation_speed = max(rotation_speed - 0.01, 0.01)  # Limit min speed

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         continue

#     # Flip the image for a mirrored display
#     image = cv2.flip(image, 1)
    
#     # Convert BGR to RGB for MediaPipe processing
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Process the image to find hands
#     results = hands.process(image_rgb)

#     if results.multi_hand_landmarks:
#         hand_landmarks = results.multi_hand_landmarks[0]
        
#         # Recognize gestures
#         recognize_gesture(hand_landmarks)

#         # Draw landmarks for debugging and user feedback
#         mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     # OpenGL rendering
#     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

#     # Rotate cube on x, y, and z axes if not grabbed
#     if not is_grabbed:
#         rotation_angles['x'] += rotation_speed  # Rotate slowly on X-axis
#         rotation_angles['y'] += rotation_speed  # Rotate slowly on Y-axis
#         rotation_angles['z'] += rotation_speed  # Rotate slowly on Z-axis

#     # Apply rotation transformations
#     glPushMatrix()
#     glTranslatef(model_pos[0], model_pos[1], model_pos[2])  # Move model based on model_pos
#     glRotatef(rotation_angles['x'], 1, 0, 0)  # Rotate around the X-axis
#     glRotatef(rotation_angles['y'], 0, 1, 0)  # Rotate around the Y-axis
#     glRotatef(rotation_angles['z'], 0, 0, 1)  # Rotate around the Z-axis
#     draw_cube()
#     glPopMatrix()

#     pygame.display.flip()
#     pygame.time.wait(10)
    
#     # Display the image
#     cv2.imshow('Hand Gesture Recognition', image)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# pygame.quit()