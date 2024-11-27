import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define indices for eye landmarks (Mediapipe FaceMesh landmarks for eyes)
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]


def midpoint(p1, p2):
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2


# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates of key points for eyes
            left_eye = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye = [face_landmarks.landmark[i] for i in RIGHT_EYE]

            # Convert normalized coordinates to pixel values
            left_eye_coords = [(int(pt.x * w), int(pt.y * h)) for pt in left_eye]
            right_eye_coords = [(int(pt.x * w), int(pt.y * h)) for pt in right_eye]

            # Find the midpoints of eyes
            left_midpoint = midpoint(left_eye_coords[0], left_eye_coords[1])
            right_midpoint = midpoint(right_eye_coords[0], right_eye_coords[1])

            # Draw eyes and gaze points
            cv2.circle(frame, left_midpoint, 5, (0, 255, 0), -1)
            cv2.circle(frame, right_midpoint, 5, (0, 255, 0), -1)

            # Simple gaze direction (relative to the eye box)
            if left_midpoint[0] < (left_eye_coords[0][0] + left_eye_coords[1][0]) // 2:
                gaze_direction = "Looking Left"
            elif left_midpoint[0] > (left_eye_coords[0][0] + left_eye_coords[1][0]) // 2:
                gaze_direction = "Looking Right"
            else:
                gaze_direction = "Looking Center"

            # Display gaze direction on the frame
            cv2.putText(frame, gaze_direction, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Eye Tracking", frame)

    # Exit loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()