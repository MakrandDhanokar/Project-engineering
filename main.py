import os
import pickle
import datetime as dt_module
from datetime import datetime as dt_class
import time
import face_recognition
import numpy as np
from flask import Flask, render_template, Response, jsonify
import cv2
from flask import session
from flask import Flask, render_template, request, jsonify, redirect
import firebase_admin
from firebase_admin import credentials, db, storage
from colorama import Fore, Style

app = Flask(__name__)

app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/Images'

# Initialize Firebase Admin SDK
cred = credentials.Certificate('E:/Learning/Deep Learning/Projects/BE final/serviceAccountKey.json')  # Replace with your service account key
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://attendance-a7527-default-rtdb.firebaseio.com/',  # Replace with your database URL
    'storageBucket': 'attendance-a7527.appspot.com'
})

camera = cv2.VideoCapture(0)

file_path = os.path.join('static', 'model', 'EncoderFile.p')
file = open(file_path, 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodedFaceKnown, studentIDs = encodeListKnownWithIds

def mark_attendance(name):
    now = dt_class.now()
    dt_string = now.strftime('%d-%B-%Y')
    folder_name = "Attendance"
    filename = os.path.join(folder_name, dt_string + '.csv')

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('Name,Time,Total Time\n')

    with open(filename, 'r') as f:
        my_data_list = f.readlines()

    name_list = [line.strip().split(',')[0] for line in my_data_list]

    dt_string = now.strftime('%H:%M:%S')
    if name not in name_list:
        dt_string = now.strftime('%H:%M:%S')
        date = dt_class.now().strftime("%Y-%m-%d")
        employee_info = db.reference(f'Employee Attendance/{name}').get()
        ref = db.reference(f'Employee Attendance/{name}')
        # employee_info['total_attendance'] += 1
        # employee_info['last_attendance_time'] = f"{date} {dt_string}"
        # ref.child('total_attendance').set(employee_info['total_attendance'])
        # ref.child('last_attendance_time').set(employee_info['last_attendance_time'])

        with open(filename, 'a') as f:
            f.write(f'{name},{dt_string},0\n')
    else:
        index = name_list.index(name)
        info = my_data_list[index].strip().split(',')
        total_time = int(info[2]) + 5
        dt_string = info[1]
        my_data_list[index] = f'{name},{dt_string},{total_time}\n'

        with open(filename, 'w') as f:
                f.writelines(my_data_list)


def encode_images(images_folder, model_folder):
    image_list = os.listdir(images_folder)
    encodings = []
    student_ids = []

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    for image_name in image_list:
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)
        student_id = os.path.splitext(image_name)[0]
        student_ids.append(student_id)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(image_rgb)[0]
        encodings.append(encoding)

        # Save encoded image to model folder
        encoded_image_path = os.path.join(model_folder, f"{student_id}.p")
        with open(encoded_image_path, 'wb') as file:
            pickle.dump(encoding, file)

    encode_list_with_ids = [encodings, student_ids]

    # Save encodings list and student IDs as a pickle file
    encoder_file_path = os.path.join(model_folder, "EncoderFile.p")
    with open(encoder_file_path, 'wb') as file:
        pickle.dump(encode_list_with_ids, file)

    print(f"Encoding Complete...")

def generate_frames():
    global encodedFaceKnown, camera

    detection_interval = 5  # Detection interval in seconds
    last_detection_time = time.time()

    while True:
        current_time = time.time()
        elapsed_time = current_time - last_detection_time

        success, frame = camera.read()
        if not success:
            break
        else:
            if elapsed_time >= detection_interval:
                imgSmall = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
                imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

                faceCurrentFrame = face_recognition.face_locations(imgSmall)
                encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

                if faceCurrentFrame:
                    for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
                        matches = face_recognition.compare_faces(encodedFaceKnown, encodeFace)
                        faceDistance = face_recognition.face_distance(encodedFaceKnown, encodeFace)

                        matchIndex = np.argmin(faceDistance)

                        if matches[matchIndex]:
                            id = studentIDs[matchIndex]
                            mark_attendance(id)

                last_detection_time = current_time

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    # Fetching employee data and processing image URLs
    ref = db.reference('Employee Attendance')  # Reference to 'Employee Attendance' node
    employee_data = ref.get()  # Fetch employee details from Firebase Realtime Database

    if employee_data:
        for employee_id, employee_details in employee_data.items():
            bucket = storage.bucket()
            blob = bucket.get_blob(f'Images/{employee_id}.jpg')
            if blob:
                expiration = dt_module.timedelta(hours=1)
                signed_url = blob.generate_signed_url(expiration=expiration)
                employee_details['image_url'] = signed_url
            else:
                print(f"Image not found for {employee_id}")
                # Handle the case where the image doesn't exist

    return render_template('index.html', employee_data=employee_data if 'employee_data' in locals() else None)


@app.route('/services')
def services():
    return render_template('services.html')
@app.route('/video_feed')
def video_feed():

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attendance_data')
def attendance_data():
    dt_string = time.strftime('%d-%B-%Y')
    filename = os.path.join("Attendance", f"{dt_string}.csv")

    attendance_data = []

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip the header line
                columns = line.strip().split(',')
                attendance_data.append({
                    'Name': columns[0],
                    'Time': columns[1],
                    'Total Time': columns[2]  # Assuming the total time is at index 2 in CSV
                })

        return jsonify({'attendance_data': attendance_data})
    else:
        return jsonify({'message': "No attendance data available for today."})


@app.route('/user_info/<employee_id>')
def info(employee_id):
    ref = db.reference('Employee Attendance')
    bucket = storage.bucket()
    blob = bucket.get_blob(f'Images/{employee_id}.jpg')
    employee_details = ref.child(employee_id).get()
    if blob:
        expiration = dt_module.timedelta(hours=1)
        signed_url = blob.generate_signed_url(expiration=expiration)
        employee_details['image_url'] = signed_url
    return render_template('employee_info.html', user_details=employee_details)

@app.route('/submit-form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        # Get form data
        name = request.form['first-name']
        id = request.form['id']
        email = request.form['email']
        address = request.form['address']
        major = request.form['major']
        starting_year = request.form['starting-year']
        password = request.form['password']
        phone = request.form['phone']
        year = request.form['year']
        dob = request.form['dob']

        # Check if employee_id exists in the form data
        if 'id' not in request.form:
            return 'Employee ID not found in form data!'

        id = request.form['id']

        # Prepare the data to be updated in Firebase
        data = {
            'name': name,
            'id': id,
            'email': email,
            'address': address,
            'major': major,
            'starting_year': starting_year,
            'password': password,
            'phone': phone,
            'total_attendance': 0,
            'year': year,
            'dob': dob
        }
        db.reference(f'Employee Attendance/{id}').set(data)

        try:
            # Check if the photo key exists in the request.files dictionary
            if 'photo' not in request.files:
                return 'No photo part in the form!'

            photo = request.files['photo']

            # Check if the file was not selected
            if photo.filename == '':
                return 'No selected photo!'

            employee_id = request.form.get('id')

            # Rename the photo to id.jpg
            photo.filename = f"{employee_id}.jpg"

            # Save the photo
            photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo.filename))
            upload_images_to_firebase(local_images_folder, firebase_images_folder)
            images_folder_path = 'static/Images'
            model_folder_path = 'static/model'
            encode_images(images_folder_path, model_folder_path)
            return 'successfully!'

        except Exception as e:
            return f"Error uploading photo: {str(e)}"


def upload_images_to_firebase(local_folder_path, firebase_folder_path):
    for filename in os.listdir(local_folder_path):
        local_file_path = os.path.join(local_folder_path, filename)
        if os.path.isfile(local_file_path):
            bucket = storage.bucket()
            blob = bucket.blob(firebase_folder_path + '/' + filename)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {filename} to Firebase Storage")

# Provide the local folder path and Firebase Storage folder path
local_images_folder = 'static/Images'
firebase_images_folder = 'Images'

if __name__ == "__main__":
    app.run(debug=True)
