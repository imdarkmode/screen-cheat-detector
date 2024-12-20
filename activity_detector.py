import random
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Whether to show real or fake numbers
LIE_TO_FRIENDS = True
SHOW_TEXT = "Eye, Heart, Temp Stimulation Tracker"
MIN_BODY_TEMP = random.uniform(96.0, 97.0)
MAX_BODY_TEMP = MIN_BODY_TEMP + random.uniform(1, 2)

MIN_HEARTRATE = random.randint(63, 87)
MAX_HEARTRATE = MIN_HEARTRATE + random.randint(4, 7)

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

    cap = cv2.VideoCapture(0)

    # Get the FPS of the video stream
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Pupil size & movement calibration started. Look at center of target area.")

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

        if not LIE_TO_FRIENDS:
            visualize_landmarks(detection_result, frame)

        if detection_result.face_landmarks:
            # Process the first detected face
            face_landmark = detection_result.face_landmarks[0]

            if detection_result.face_landmarks:
                # Process the first detected face
                face_landmark = detection_result.face_landmarks[0]

                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key

                    nose_tip = face_landmark[NOSE_TIP_INDEX]
                    pupil_avg_y = (face_landmark[LEFT_EYE_PUPIL_INDEX].y + face_landmark[RIGHT_EYE_PUPIL_INDEX].y) / 2
                    eye_avg_y = (face_landmark[LEFT_EYE_INNER_INDEX].y + face_landmark[LEFT_EYE_OUTER_INDEX].y +
                                 face_landmark[RIGHT_EYE_INNER_INDEX].y +
                                 face_landmark[RIGHT_EYE_OUTER_INDEX].y) / 4

                    # Compute the head pose (roll, pitch, yaw) using the relative positions of the landmarks
                    pitch = calculate_head_pose(nose_tip, face_landmark[LEFT_EYE_INNER_INDEX])

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
                        "pupil_distance_to_nose_tip": round(pupil_avg_y - nose_tip.y, decimal_places_rounding),
                        "eye_distance_to_nose_tip": round(eye_avg_y - nose_tip.y, decimal_places_rounding),
                        "pitch": pitch
                    }

                    if not LIE_TO_FRIENDS:
                        print(face_calibration)

                    return face_calibration

        # Display the full-screen video
        cv2.imshow(f"{SHOW_TEXT}", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def calculate_head_pose(nose_tip, left_eye_inner):
    """
    Calculate the head pose (roll, pitch, yaw) based on the 3D positions of nose tip and pupils.
    """
    # Compute vectors
    nose_to_left_eye = np.subtract([left_eye_inner.x, left_eye_inner.y, left_eye_inner.z], [nose_tip.x, nose_tip.y, nose_tip.z])

    # Pitch (tilting head up or down) can be estimated by comparing vertical movement
    pitch = np.arctan2(nose_to_left_eye[1], np.sqrt(nose_to_left_eye[0]**2 + nose_to_left_eye[2]**2))

    return round(pitch, decimal_places_rounding)


def adjust_distance(pupil_to_nose_distance: float, current_pitch: float, calibration_pitch: float) -> float:
    # Calculate the pitch difference
    delta_pitch = current_pitch - calibration_pitch

    # Adjust the distance using the cosine of the pitch difference
    adjusted_distance = pupil_to_nose_distance * np.cos(delta_pitch)

    return adjusted_distance


def main(calibration, user_screen):
    num_frames_looking_each_direction = {
        "Up": 0,
        "Down": 0,
        "Straight": 0
    }
    current_heartrate = 75
    current_body_temp = 98
    current_eye_dilation = 'medium'

    cap = cv2.VideoCapture(0)

    # Get the FPS of the video stream
    fps = cap.get(cv2.CAP_PROP_FPS)

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

        if not LIE_TO_FRIENDS:
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

            current_body_temp, current_eye_dilation, current_heartrate = generate_fake_numbers(current_body_temp, current_heartrate, current_eye_dilation)

            if LIE_TO_FRIENDS:
                cv2.putText(frame, f"Body Temp: {current_body_temp}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Heart Rate: {current_heartrate}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Pupil Dilation: {current_eye_dilation}", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if not LIE_TO_FRIENDS:
                cv2.putText(frame, f"Straight: {round(num_frames_looking_each_direction['Straight'] / 30, 1)}s", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Up: {round(num_frames_looking_each_direction['Up'] / 30, 1)}s", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, f"Down: {round(num_frames_looking_each_direction['Down'] / 30, 1)}s", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Display the full-screen video
        cv2.imshow(f"{SHOW_TEXT}", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print the results
    top_text = 'NP Value' if user_screen == 'top' else 'P Value'
    bottom_text = 'NP Value' if user_screen == 'bottom' else 'P Value'

    print(f"S: {round(num_frames_looking_each_direction['Straight'] / 30, 1)}s")
    print(f"{top_text}: {round(num_frames_looking_each_direction['Up'] / 30, 1)}s")
    print(f"{bottom_text}: {round(num_frames_looking_each_direction['Down'] / 30, 1)}s")

    cap.release()
    cv2.destroyAllWindows()


def determine_gaze(face_landmark, calibration, frame):
    """
    Determine if the user is looking up, down, or straight based on eye and nose positions.
    """
    nose_tip = face_landmark[NOSE_TIP_INDEX]
    left_eye_pupil = face_landmark[LEFT_EYE_PUPIL_INDEX]  # Left eye
    left_eye_inner = face_landmark[LEFT_EYE_INNER_INDEX]  # Left eye
    left_eye_outer_y = face_landmark[LEFT_EYE_OUTER_INDEX].y  # Left eye
    right_eye_pupil = face_landmark[RIGHT_EYE_PUPIL_INDEX]  # Right eye
    right_eye_inner_y = face_landmark[RIGHT_EYE_INNER_INDEX].y  # Right eye
    right_eye_outer_y = face_landmark[RIGHT_EYE_OUTER_INDEX].y  # Right eye

    # Calculate the average pupil level
    pupil_avg_y = (left_eye_pupil.y + right_eye_pupil.y) / 2
    pupil_distance_to_nose_tip = pupil_avg_y - nose_tip.y
    pupil_distance_to_nose_tip = round(pupil_distance_to_nose_tip, decimal_places_rounding)

    # Calculate the average inner/outer eye level
    eye_avg_y = (left_eye_inner.y + left_eye_outer_y + right_eye_inner_y + right_eye_outer_y) / 4
    eye_distance_to_nose_tip = eye_avg_y - nose_tip.y
    eye_distance_to_nose_tip = round(eye_distance_to_nose_tip, decimal_places_rounding)

    # Normalize the distances by considering the head pose (roll and pitch) from calibration
    calib_pitch = calibration['pitch']

    # Compute the head pose (roll, pitch, yaw) using the relative positions of the landmarks
    current_pitch = calculate_head_pose(nose_tip, left_eye_inner)

    # Normalize based on pitch (vertical tilt)
    adjusted_pupil_distance = adjust_distance(pupil_distance_to_nose_tip, current_pitch, calib_pitch)
    adjusted_eye_distance = adjust_distance(eye_distance_to_nose_tip, current_pitch, calib_pitch)

    # Calculate the difference between the eye and pupil from calibration (this will be considered straight)
    calib_diff = calibration['pupil_distance_to_nose_tip'] - calibration['eye_distance_to_nose_tip']
    current_diff = adjusted_pupil_distance - adjusted_eye_distance

    total_diff = round(current_diff - calib_diff, decimal_places_rounding)

    # Print current total difference
    if not LIE_TO_FRIENDS:
        cv2.putText(frame, f"current pitch: {current_pitch}", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2)

    # Determine gaze direction
    if total_diff > 0.0004:
        return -1
    if total_diff < -0.0004:
        return 1
    else:
        return 0


def generate_fake_numbers(current_body_temp: float, current_heartrate: int, current_eye_dilation: str):
    # Body temp
    if random.random() < 0.05:
        body_temp = round(
            max(MIN_BODY_TEMP, min(MAX_BODY_TEMP, current_body_temp + random.uniform(-0.1, 0.1))), 1
        )
    else:
        body_temp = current_body_temp

    # Eye dilation
    if random.random() < 0.005:
        possible_dilations = ['medium', 'high', 'low']
        possible_dilations.remove(current_eye_dilation)  # Exclude the current state
        eye_dilation = random.choice(possible_dilations)  # Pick a new state
    else:
        eye_dilation = current_eye_dilation

    # Heart rate
    if random.random() < 0.03:
        heartrate = max(MIN_HEARTRATE, min(MAX_HEARTRATE, current_heartrate + random.randint(-1, 1)))
    else:
        heartrate = current_heartrate

    return body_temp, eye_dilation, heartrate


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
    screen = 'top'

    if not LIE_TO_FRIENDS:
        user_input = input("Top Screen? (y/n):")
    else:
        user_input = input("Press a button to Start:")

    if user_input != 'y':
        screen = 'bottom'

    calibration_results = calibration_step()
    main(calibration_results, screen)
