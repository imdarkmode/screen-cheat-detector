import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Mediapipe constants
LEFT_EYE_INNER_INDEX = 133
LEFT_EYE_PUPIL_INDEX = 468
LEFT_EYE_OUTER_INDEX = 33

# Right eye
RIGHT_EYE_INNER_INDEX = 362
RIGHT_EYE_PUPIL_INDEX = 473
RIGHT_EYE_OUTER_INDEX = 359

# Nose tip
NOSE_TIP_INDEX = 1  # Nose tip for reference

decimal_places_rounding = 4

ALL_LANDMARKS = [LEFT_EYE_PUPIL_INDEX, LEFT_EYE_INNER_INDEX, LEFT_EYE_OUTER_INDEX, RIGHT_EYE_PUPIL_INDEX,
                 RIGHT_EYE_INNER_INDEX, RIGHT_EYE_OUTER_INDEX, NOSE_TIP_INDEX]

# Path to the Mediapipe Face Landmarker model
MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"

# Configure and create the FaceLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,  # Disable blend shapes for better performance
    output_facial_transformation_matrixes=False,  # Disable 3D matrices
    num_faces=1  # Detect one face at a time
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)


def calibration_step():

    cap = cv2.VideoCapture(1)

    # Get the FPS of the video stream
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Calibration started. Look at the center of the screen and hit enter.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert to RGB format for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = face_landmarker.detect(mp_image)

        visualize_landmarks(detection_result, frame)

        if detection_result.face_landmarks:
            # Process the first detected face
            face_landmark = detection_result.face_landmarks[0]

            if detection_result.face_landmarks:
                # Process the first detected face
                face_landmark = detection_result.face_landmarks[0]

                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key

                    nose_tip_y = face_landmark[NOSE_TIP_INDEX].y
                    pupil_avg_y = (face_landmark[LEFT_EYE_PUPIL_INDEX].y + face_landmark[RIGHT_EYE_PUPIL_INDEX].y) / 2
                    eye_avg_y = (face_landmark[LEFT_EYE_INNER_INDEX].y + face_landmark[LEFT_EYE_OUTER_INDEX].y +
                                 face_landmark[RIGHT_EYE_INNER_INDEX].y +
                                 face_landmark[RIGHT_EYE_OUTER_INDEX].y) / 4

                    # Compute the head pose (roll, pitch, yaw) using the relative positions of the landmarks
                    roll, pitch, yaw = calculate_head_pose(nose_tip_y, face_landmark[LEFT_EYE_INNER_INDEX].y, face_landmark[RIGHT_EYE_INNER_INDEX].y)

                    face_calibration = {
                        "nose_tip_y": face_landmark[NOSE_TIP_INDEX].y,
                        "left_eye_pupil_y": face_landmark[LEFT_EYE_PUPIL_INDEX].y,  # Left eye,
                        "left_eye_inner_y": face_landmark[LEFT_EYE_INNER_INDEX].y,  # Left eye
                        "left_eye_outer_y": face_landmark[LEFT_EYE_OUTER_INDEX].y,  # Left eye
                        "right_eye_pupil_y": face_landmark[RIGHT_EYE_PUPIL_INDEX].y,  # Right eye
                        "right_eye_inner_y": face_landmark[RIGHT_EYE_INNER_INDEX].y,  # Right eye
                        "right_eye_outer_y": face_landmark[RIGHT_EYE_OUTER_INDEX].y,  # Right eye
                        "pupil_avg_y": pupil_avg_y,
                        "eye_avg_y": eye_avg_y,
                        "pupil_distance_to_nose_tip": round(pupil_avg_y - nose_tip_y, decimal_places_rounding),
                        "eye_distance_to_nose_tip": round(eye_avg_y - nose_tip_y, decimal_places_rounding),
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw,
                    }

                    print(face_calibration)

                    return face_calibration

        # Display the full-screen video
        cv2.imshow("Simplified Gaze Tracking", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def calculate_head_pose(nose_tip_y, left_eye_inner_y, right_eye_inner_y):
    """
    Calculate the head pose (roll, pitch, yaw) based on the 3D positions of nose tip and pupils.
    """
    # Compute vectors
    nose_to_left_eye = left_eye_inner_y - nose_tip_y
    nose_to_right_eye = right_eye_inner_y - nose_tip_y

    # Calculate the angle between these two vectors using dot product
    dot_product = np.dot(nose_to_left_eye, nose_to_right_eye)
    norm_left_eye = np.linalg.norm(nose_to_left_eye)
    norm_right_eye = np.linalg.norm(nose_to_right_eye)

    # Compute the angle between the two vectors
    angle_between_eyes = np.arccos(dot_product / (norm_left_eye * norm_right_eye))

    # Roll is based on the angle between the eyes, assuming a simple case for demonstration
    roll = np.degrees(angle_between_eyes)

    # Pitch (tilting head up or down) can be estimated by comparing vertical movement
    pitch = np.arctan2(nose_to_left_eye[2], norm_left_eye)  # Using the 3D nose to left eye vector

    # Yaw (left-right head rotation) can be calculated similarly based on horizontal differences
    yaw = np.arctan2(nose_to_left_eye[1], norm_left_eye)

    return roll, pitch, yaw

def main(calibration):
    num_frames_looking_each_direction = {
        "Up": 0,
        "Down": 0,
        "Straight": 0
    }

    cap = cv2.VideoCapture(1)

    # Get the FPS of the video stream
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Frames per second: {fps}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert to RGB format for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = face_landmarker.detect(mp_image)

        visualize_landmarks(detection_result, frame)

        if detection_result.face_landmarks:
            # Process the first detected face
            face_landmark = detection_result.face_landmarks[0]

            # Get eye and nose positions
            gaze_direction = determine_gaze(face_landmark, calibration, frame)
            gaze_text = ""

            if gaze_direction == 0:
                gaze_text = "Straight"
                num_frames_looking_each_direction['Straight'] += 1
            if gaze_direction == 1:
                gaze_text = "Up"
                num_frames_looking_each_direction['Up'] += 1
            if gaze_direction == -1:
                gaze_text = "Down"
                num_frames_looking_each_direction['Down'] += 1

            # Display status text
            cv2.putText(frame, gaze_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Display num frames
            cv2.putText(frame, f"Straight: {round(num_frames_looking_each_direction['Straight'] / 30, 1)}s", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"Up: {round(num_frames_looking_each_direction['Up'] / 30, 1)}s", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"Down: {round(num_frames_looking_each_direction['Down'] / 30, 1)}s", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the full-screen video
        cv2.imshow("Simplified Gaze Tracking", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print the results
    print(f"Straight: {round(num_frames_looking_each_direction['Straight'] / 30, 1)}s")
    print(f"Up: {round(num_frames_looking_each_direction['Up'] / 30, 1)}s")
    print(f"Down: {round(num_frames_looking_each_direction['Down'] / 30, 1)}s")

    cap.release()
    cv2.destroyAllWindows()


def determine_gaze(face_landmark, calibration, frame):
    """
    Determine if the user is looking up, down, or straight based on eye and nose positions.
    """
    nose_tip = face_landmark[NOSE_TIP_INDEX]
    left_eye_pupil = face_landmark[LEFT_EYE_PUPIL_INDEX]  # Left eye
    left_eye_inner_y = face_landmark[LEFT_EYE_INNER_INDEX].y  # Left eye
    left_eye_outer_y = face_landmark[LEFT_EYE_OUTER_INDEX].y  # Left eye
    right_eye_pupil = face_landmark[RIGHT_EYE_PUPIL_INDEX]  # Right eye
    right_eye_inner_y = face_landmark[RIGHT_EYE_INNER_INDEX].y  # Right eye
    right_eye_outer_y = face_landmark[RIGHT_EYE_OUTER_INDEX].y  # Right eye

    # Calculate the average pupil level
    pupil_avg_y = (left_eye_pupil.y + right_eye_pupil.y) / 2
    pupil_distance_to_nose_tip = pupil_avg_y - nose_tip.y
    pupil_distance_to_nose_tip = round(pupil_distance_to_nose_tip, decimal_places_rounding)

    # Calculate the average inner/outer eye level
    eye_avg_y = (left_eye_inner_y + left_eye_outer_y + right_eye_inner_y + right_eye_outer_y) / 4
    eye_distance_to_nose_tip = eye_avg_y - nose_tip.y
    eye_distance_to_nose_tip = round(eye_distance_to_nose_tip, decimal_places_rounding)

    # Normalize the distances by considering the head pose (roll and pitch) from calibration
    calib_roll = calibration['roll']
    calib_pitch = calibration['pitch']

    # Compute the head pose (roll, pitch, yaw) using the relative positions of the landmarks
    current_roll, current_pitch, current_yaw = calculate_head_pose(nose_tip, left_eye_pupil, right_eye_pupil)

    # Normalize based on pitch (vertical tilt)
    normalized_pupil_distance = pupil_distance_to_nose_tip - (current_pitch - calib_pitch) * 0.1
    normalized_eye_distance = pupil_distance_to_nose_tip - (current_pitch - calib_pitch) * 0.1

    # Calculate the difference between the eye and pupil from calibration (this will be considered straight)
    calib_diff = calibration['pupil_distance_to_nose_tip'] - calibration['eye_distance_to_nose_tip']
    current_diff = normalized_pupil_distance - normalized_eye_distance

    total_diff = round(current_diff - calib_diff, decimal_places_rounding)

    # Print current total difference
    cv2.putText(frame, f"current pitch: {current_pitch}", (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)

    # Determine gaze direction
    if total_diff > 0.0004:
        return -1
    if total_diff < -0.0004:
        return 1
    else:
        return 0


def visualize_landmarks(detection_result, frame):
    if detection_result.face_landmarks:
        for face_landmark in detection_result.face_landmarks:
            for idx, landmark in enumerate(face_landmark):
                if idx in ALL_LANDMARKS:
                    color = (0, 0, 255)
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, color, -1)


if __name__ == "__main__":
    calibration_results = calibration_step()
    main(calibration_results)
