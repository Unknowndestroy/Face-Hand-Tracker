import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error has been occured while opening camera.")
        break


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results_hands = hands.process(image)
    results_face = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=5, circle_radius=5),
                                       mp_drawing.DrawingSpec(color=(165, 42, 42), thickness=2, circle_radius=2))


    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            h, w, _ = image.shape
            x_min = int(face_landmarks.landmark[1].x * w)
            x_max = int(face_landmarks.landmark[4].x * w)
            y_min = int(face_landmarks.landmark[152].y * h)
            y_max = int(face_landmarks.landmark[10].y * h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)


            mood = ""  
            age = ""      

            cv2.putText(image, f" {mood}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f" {age}", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("Webcam", image)
    if cv2.waitKey(5) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
