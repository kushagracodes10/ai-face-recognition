import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_detection = mp.solutions.face_detection

# Function to get simple "face embedding" using color histogram
def get_face_embedding(image, bbox):
    x, y, w, h = bbox
    # Ensure bbox is inside the image
    x, y = max(0, x), max(0, y)
    w, h = max(1, w), max(1, h)
    face = image[y:y+h, x:x+w]
    if face.size == 0:
        return None
    face = cv2.resize(face, (100, 100))
    hist = cv2.calcHist([face], [0, 1, 2], None, [8,8,8], [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Load known faces from 'known_faces' folder
known_faces = {}
if os.path.exists("known_faces"):
    for filename in os.listdir("known_faces"):
        if filename.endswith((".jpg", ".png")):
            name = os.path.splitext(filename)[0]
            image = cv2.imread(f"known_faces/{filename}")
            if image is None:
                continue
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
                results = face_detection.process(rgb_image)
                if results.detections:
                    d = results.detections[0].location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    bbox = int(d.xmin*iw), int(d.ymin*ih), int(d.width*iw), int(d.height*ih)
                    embedding = get_face_embedding(image, bbox)
                    if embedding is not None:
                        known_faces[name] = embedding

# Start webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed, retrying...")
            continue  # don’t break, just retry

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin*iw), int(bboxC.ymin*ih), int(bboxC.width*iw), int(bboxC.height*ih)

                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Compute embedding
                embedding = get_face_embedding(frame, (x, y, w, h))
                name = "Unknown"

                if embedding is not None:
                    min_dist = float("inf")
                    for k_name, k_emb in known_faces.items():
                        dist = np.linalg.norm(embedding - k_emb)
                        if dist < min_dist and dist < 0.5:  # threshold
                            min_dist = dist
                            name = k_name

                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # Exit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
