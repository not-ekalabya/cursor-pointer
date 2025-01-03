import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)
webcam_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
webcam_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Disable pyautogui's failsafe
pyautogui.FAILSAFE = False

def map_coordinates(x, y, input_width, input_height):
    # Map the coordinates from webcam space to screen space
    screen_x = np.interp(x, [0, input_width], [0, screen_width])
    screen_y = np.interp(y, [0, input_height], [0, screen_height])
    return screen_x, screen_y

def get_calibration_point(cap, pose, prompt):
    print(f"Move your left hand {prompt} and hold still, then press SPACE")
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

        cv2.putText(image, f"Move hand {prompt}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Calibration', image)

        key = cv2.waitKey(5) & 0xFF
        if key == ord(' '):  # Space key
            if results.pose_landmarks:
                wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                return (wrist.x * webcam_width, wrist.y * webcam_height)
        elif key == ord('q'):
            return None

def calibrate():
    calibration_points = {}
    prompts = ['to the RIGHT', 'to the LEFT', 'to the TOP', 'to the BOTTOM']
    positions = ['right', 'left', 'top', 'bottom']
    
    for prompt, position in zip(prompts, positions):
        point = get_calibration_point(cap, pose, prompt)
        if point is None:  # User quit
            return None
        calibration_points[position] = point
    
    return calibration_points

def map_coordinates_calibrated(x, y, calibration_points):
    # Map coordinates based on calibration points, allowing full range of motion
    x_min = calibration_points['left'][0]
    x_max = calibration_points['right'][0]
    y_min = calibration_points['top'][1]
    y_max = calibration_points['bottom'][1]
    
    # Clamp input values to calibration boundaries
    x = max(min(x, x_max), x_min)
    y = max(min(y, y_max), y_min)
    
    screen_x = np.interp(x, [x_min, x_max], [0, screen_width])
    screen_y = np.interp(y, [y_min, y_max], [0, screen_height])
    return screen_x, screen_y

# Get calibration points before starting main loop
print("Starting calibration...")
calibration_points = calibrate()

if calibration_points:
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                wrist_x = wrist.x * webcam_width
                wrist_y = wrist.y * webcam_height

                # Use calibrated mapping for continuous movement
                screen_x, screen_y = map_coordinates_calibrated(
                    wrist_x, wrist_y, calibration_points)

                pyautogui.moveTo(screen_x, screen_y, duration=0.1)

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Hand Tracker', image)

            # Check for escape key (27 is the ASCII code for Escape)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
