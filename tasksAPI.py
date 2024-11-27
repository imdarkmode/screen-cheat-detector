import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Important landmarks
LEFT_EYE_INNER_INDEX = 133
LEFT_PUPIL_CENTER_INDEX = 468
LEFT_EYE_OUTER_INDEX = 33

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
faceLandmarker = vision.FaceLandmarker.create_from_options(options)


def start_calibration():
    calibration_vectors = []

    print("Calibration started. Look at the top left and hit Enter.")
    cap = cv2.VideoCapture(0)
    calibration_step = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        detection_result = faceLandmarker.detect(mp_image)

        # Visualize landmarks and calibration points
        visualize_landmarks(detection_result, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            calibration_vector = get_calibration_vector(detection_result)
            if calibration_vector is not None:
                calibration_vectors.append(calibration_vector)
                calibration_step += 1
                print(f"Calibration step {calibration_step} recorded.")
                if calibration_step == 4:
                    break
                else:
                    prompts = [
                        "top right",
                        "bottom right",
                        "bottom left"
                    ]
                    print(f"Now look at the {prompts[calibration_step - 1]} and hit Enter.")
            else:
                print("Failed to detect gaze direction. Please try again.")

        # Show the video with landmarks and gaze direction
        cv2.imshow("Calibration - Gaze Tracking", frame)

        # Exit when 'q' is pressed
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    if len(calibration_vectors) == 4:
        print("Calibration complete!")
        return calibration_vectors
    else:
        print("Calibration failed. Please try again.")
        return None


def start_gaze_tracking(calibration_vectors):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        detection_result = faceLandmarker.detect(mp_image)
        visualize_landmarks(detection_result, frame)

        # Calculate the gaze vector
        current_vector = get_calibration_vector(detection_result)
        if current_vector is not None:
            within_bounds = is_gaze_within_bounds(current_vector, calibration_vectors)
            status_text = "Gaze within bounds!" if within_bounds else "Gaze outside!"
            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the video with landmarks
        cv2.imshow("Gaze Tracking", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def visualize_landmarks(detection_result, frame):
    if detection_result.face_landmarks:
        for face_landmark in detection_result.face_landmarks:
            for idx, landmark in enumerate(face_landmark):
                if idx in [LEFT_PUPIL_CENTER_INDEX, LEFT_EYE_INNER_INDEX, LEFT_EYE_OUTER_INDEX]:
                    color = (0, 0, 255)
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, color, -1)


def get_calibration_vector(detection_results):
    if detection_results.face_landmarks:
        for face_landmark in detection_results.face_landmarks:
            eye_center = np.array([face_landmark[LEFT_PUPIL_CENTER_INDEX].x,
                                   face_landmark[LEFT_PUPIL_CENTER_INDEX].y,
                                   face_landmark[LEFT_PUPIL_CENTER_INDEX].z])
            inner_corner = np.array([face_landmark[LEFT_EYE_INNER_INDEX].x,
                                     face_landmark[LEFT_EYE_INNER_INDEX].y,
                                     face_landmark[LEFT_EYE_INNER_INDEX].z])
            outer_corner = np.array([face_landmark[LEFT_EYE_OUTER_INDEX].x,
                                     face_landmark[LEFT_EYE_OUTER_INDEX].y,
                                     face_landmark[LEFT_EYE_OUTER_INDEX].z])
            midpoint = (inner_corner + outer_corner) / 2
            gaze_vector = midpoint - eye_center
            gaze_vector /= np.linalg.norm(gaze_vector)
            return gaze_vector
    return None


def is_gaze_within_bounds(current_vector, calibration_vectors, threshold=0.9):
    for calib_vector in calibration_vectors:
        similarity = np.dot(current_vector, calib_vector)
        if similarity >= threshold:
            return True
    return False


if __name__ == "__main__":
    vectors = start_calibration()
    if vectors:
        start_gaze_tracking(vectors)
    else:
        print("Calibration failed. Exiting.")
