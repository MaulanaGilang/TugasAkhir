# Import necessary libraries
import numpy as np
from keras.models import load_model
# Additional imports
import copy

def preprocess_landmarks(landmarks):
    # Flatten the landmark points into a list or an array
    # Make sure to only take the first 468 landmarks if that's the expected number
    flattened_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark[:468]]).flatten()
    expected_size = 468 * 3  # Adjust this value if the model expects a different number of landmarks
    if len(flattened_landmarks) != expected_size:
        raise ValueError(f"Expected {expected_size} values, but got {len(flattened_landmarks)}")
    # Reshape the landmarks as needed for your model input
    processed_landmarks = flattened_landmarks.reshape(-1, expected_size)
    return processed_landmarks

def cropped_image_normal(frame, landmarks):
    # Calculate the bounding box around the landmarks
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    H, W, _ = frame.shape
    x_min = int(x_min * W)
    x_max = int(x_max * W)
    y_min = int(y_min * H)
    y_max = int(y_max * H)

    # Find the longest side to make the bounding box a square
    width = x_max - x_min
    height = y_max - y_min
    longest_side = max(width, height)

    # Calculate new x_min, x_max, y_min, y_max to make the crop a square
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_min = max(0, x_center - longest_side // 2)
    x_max = min(W, x_center + longest_side // 2)
    y_min = max(0, y_center - longest_side // 2)
    y_max = min(H, y_center + longest_side // 2)

    # Crop the image
    cropped_image = frame[y_min:y_max, x_min:x_max]
    return cropped_image

def cropped_image_black(frame, landmarks):
    # Calculate the bounding box around the landmarks
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    H, W, _ = frame.shape
    x_min = int(x_min * W)
    x_max = int(x_max * W)
    y_min = int(y_min * H)
    y_max = int(y_max * H)

    # Find the longest side to make the bounding box a square
    width = x_max - x_min
    height = y_max - y_min
    longest_side = max(width, height)

    # Calculate new x_min, x_max, y_min, y_max to make the crop a square
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    x_min = max(0, x_center - longest_side // 2)
    x_max = min(W, x_center + longest_side // 2)
    y_min = max(0, y_center - longest_side // 2)
    y_max = min(H, y_center + longest_side // 2)

    # Crop the image
    cropped_black = frame[y_min:y_max, x_min:x_max]

    return cropped_black

def Klasifikasi(Image, ModelCNN):
    X = []
    img = copy.deepcopy(Image)
    img = cv2.resize(img, (200, 200))
    img = np.asarray(img) / 255
    X.append(img)
    X = np.array(X)
    X = X.astype('float32')
    hs = ModelCNN.predict(X, verbose=0) #sebelum manggil predict dijadiin item sm crop

    idx = -1
    if hs.max() > 0.5:
        idx = np.argmax(hs)
        print("Raw predictions:", hs)
    return idx

def PredictFaceMesh(NoKamera, LabelKelas):
    ModelCNN = load_model('D:/1 PraTA/Code/CNN/model.h5')
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(NoKamera)

    PrevIdx = -1  # Define PrevIdx before use
    counter = 0  # Define counter before use

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Create a black image with the same dimensions as the frame
            black_image = np.zeros_like(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face landmarks on the original image
                    mp_drawing.draw_landmarks(
                        image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3)
                    )

                    # Draw the face landmarks on the black image
                    mp_drawing.draw_landmarks(
                        black_image,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=3)
                    )
                            # Draw specific landmarks with custom colors
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                    
                        # if idx == 4: 
                        #     color = (255, 0, 0)
                        #     cv2.circle(image, (x, y), 5, color, -1)
                        #     cv2.circle(black_image, (x, y), 5, color, -1)
                        # elif idx == 152: 
                        #     color = (0, 0, 255)
                        #     cv2.circle(image, (x, y), 5, color, -1)
                        #     cv2.circle(black_image, (x, y), 5, color, -1)
                        # elif idx == 10:
                        #     color = (0, 255, 0)
                        #     cv2.circle(image, (x, y), 5, color, -1)
                        #     cv2.circle(black_image, (x, y), 5, color, -1)
                        # elif idx ==234:
                        #     color = (255, 0, 255)
                        #     cv2.circle(image, (x, y), 5, color, -1)
                        #     cv2.circle(black_image, (x, y), 5, color, -1)
                        # elif idx == 454:
                        #     color = (255, 255, 0) #bgr
                        #     cv2.circle(image, (x, y), 5, color, -1)
                        #     cv2.circle(black_image, (x, y), 5, color, -1)


                    # Assuming classify_landmarks is the correct function to call
                    # and that it internally calls preprocess_landmarks
                    predicted_label = preprocess_landmarks(face_landmarks)
                    cropped_black = cropped_image_black(image, face_landmarks)

                    idx = Klasifikasi(cropped_black, ModelCNN)
                    x = 50
                    y = 50
                    
                    if idx == 0 and idx != PrevIdx:
                        counter += 1
                        PrevIdx = idx
 
                    print("Index:", idx)
                    if idx >= 0:
                        print("Drawing label:", LabelKelas[idx])
                        cv2.putText(image, LabelKelas[idx], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 3)
                        cv2.putText(black_image, LabelKelas[idx], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 0), 3)
                        # Display the black image with landmarks
            cv2.imshow('Black Image with Landmarks', black_image)

            cv2.imshow('MediaPipe Face Mesh', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

# Define class labels for classification
LabelKelas = ["airgap", "noairgap"]