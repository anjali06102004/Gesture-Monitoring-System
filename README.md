# Gesture-Monitoring-System
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Video Capture
cap = cv2.VideoCapture(0)

def classify_pose(landmarks):
    """
    Classify the pose based on keypoints.
    """
    # Get coordinates
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the mid-point of hips and shoulders
    mid_hip = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])
    mid_shoulder = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])

    # Calculate the vertical distance between hips and shoulders
    vertical_distance = np.linalg.norm(mid_hip - mid_shoulder)

    # Determine activity based on vertical distance
    if vertical_distance < 0.2:
        return "Laying Down"
    elif mid_hip[1] < mid_shoulder[1]:
        return "Standing"
    else:
        return "Sitting"
    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Classify pose
        landmarks = results.pose_landmarks.landmark
        activity = classify_pose(landmarks)

        # Display the activity on the frame
        cv2.putText(frame, activity, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Human Activity Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
