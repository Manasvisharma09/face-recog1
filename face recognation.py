import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

def load_encode_faces(directory):
    known_face_encodings = []
    known_face_names = []
 
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png", ".jpeg", ".jfif")):
            # Load the image file
            image_path = os.path.join(directory, filename)
            face_image = face_recognition.load_image_file(image_path)

            # Convert image to RGB format
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Compute face encodings
            face_locations = face_recognition.face_locations(face_image)
            face_encodings = face_recognition.face_encodings(face_image, face_locations)
            if len(face_encodings) > 0:
                known_face_encodings.append(face_encodings[0])  # Use the first face encoding
                known_face_names.append(os.path.splitext(filename)[0])  # Use the filename as the name

    return known_face_encodings, known_face_names

# Directory containing face images
faces_directory = "faces"

# Load and encode faces from the directory
known_face_encodings, known_face_names = load_encode_faces(faces_directory)

# Test printing loaded face names
print(known_face_names)
students = known_face_names.copy()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Compute face encodings for the detected face locations
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        # Compare face encodings with known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""
        
        # Calculate face distances to find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # Assign the name if the match is found
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        
        # If the name is in known face names and students, remove it and log attendance
        if name in known_face_names and name in students:
            
        
            students.remove(name)
            print(students)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])

    cv2.imshow("attendance system", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        exit(0);

video_capture.release()
cv2.destroyAllWindows()
f.close()
