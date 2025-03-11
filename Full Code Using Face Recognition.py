import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2,os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import threading
import matplotlib.pyplot as plt

total_present = 0  

# Function to capture user's face during registration
def capture_face(id, name):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    image_folder = "face_images"
    os.makedirs(image_folder, exist_ok=True)

    start_time = time.time()
    count = 0

    while count < 100:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save captured face images
            cv2.imwrite(f"{image_folder}/{id}_{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('Capture Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q')or time.time() - start_time > 10:
            break

    camera.release()
    cv2.destroyAllWindows()

# Function to train LBPH recognizer using captured face images
def train_recognizer():   
    # Initialize LBPH recognizer
    recognizer = cv2.face_LBPHFaceRecognizer.create()    
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    face_images_dir = "face_images"

    # Lists to store faces and corresponding IDs
    faces = []
    ids = []

    # Load captured face images
    for root, dirs, files in os.walk(face_images_dir):
        for file in files:
            # Read image in grayscale
            path = os.path.join(root, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Extract ID from file name
                id = int(os.path.basename(path).split('_')[0])
                faces.append(img)
                ids.append(id)

    # Train recognizer
    recognizer.train(faces, np.array(ids))

    # Save trained recognizer
    recognizer.save('Trainner/recognizer.yml')


# Function to recognize face during login
def recognize_face():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read("Trainner/recognizer.yml")
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id, confidence = recognizer.predict(roi_gray)
            if confidence < 70:  # Confidence threshold
                # Fetch user details from CSV based on ID
                name = "Unknown"
                timestamp = "N/A"
                with open("users.csv", "r") as file:
                    for row in file:
                        data = row.split(",")
                        if int(data[0]) == id:
                            name = data[1]
                            timestamp = data[2]
                            break
                cv2.putText(frame, f"ID: {id} | Name: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                #cv2.putText(frame, f"Confidence: {confidence}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Recognize Face', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

# Modify registration function to include face capture
def register_new_user():
    id = id_entry.get()
    name = name_entry.get()
    if id and name:
        if not check_user_exists(id):
            save_new_user_info(id, name)
            # Capture user's face
            capture_face(id, name)
            # Train recognizer using captured faces
            train_recognizer()

            mess.showinfo("Registration Successful", "New user registered successfully!")

        else:
            mess.showerror("Registration Error", "User already exists!")
    else:
        mess.showerror("Registration Error", "Please enter both ID and Name.")

# Modify login function to include face recognition
def login():
    global total_present
    id = id_entry.get()
    if id:
        if check_user_exists(id):
            recognize_face()
            mess.showinfo("Login Successful", "Welcome back!")
            total_present += 1  # Increment total_present count
            clear_entries()
            # Plot bar graph of total summary
            plot_summary()
            # Update attendance report
            update_attendance_report(id)
        else:
            mess.showerror("Login Error", "User does not exist!")
    else:
        mess.showerror("Login Error", "Please enter ID.")


# Function to save new user information to CSV file
def save_new_user_info(id, name, filename="users.csv"):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([id, name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    clear_entries()

# Function to check if a user exists in the CSV file
def check_user_exists(id, filename="users.csv"):
    if not os.path.isfile(filename):
        return False
    with open(filename, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == id:
                return True
    return False


# Function to clear entry fields
def clear_entries():
    id_entry.delete(0, tk.END)
    name_entry.delete(0, tk.END)

# Function to plot bar graph of total summary
def plot_summary():
    plt.clf()  # Clear the previous plot
    filename = "users.csv"
    total_registered = sum(1 for line in open(filename))  # Subtract 1 to exclude header
    total_absent = total_registered - total_present

    # Plot bar graph
    labels = ['Total Registered', 'Present', 'Absent']
    values = [total_registered, total_present, total_absent]
    plt.bar(labels, values, color=['blue', 'green', 'red'])
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.title('Total Summary of Students')
    plt.tight_layout()  # Adjust layout to fit the frame
    plt.savefig('plot.png')  # Save the plot as an image
    plt.close()  # Close the plot to release resources

    # Display the plot inside frame2
    graph_label.config(image=None)
    graph_label.img = tk.PhotoImage(file='plot.png')
    graph_label.config(image=graph_label.img)

# Function to update the attendance report
def update_attendance_report(student_id):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("users.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == student_id:
                student_name = row[1]
                break
        else:
            student_name = "N/A"
    tv.insert('', 'end', values=(student_id, student_name, timestamp))

# Function to update the clock
def tick():
    # Get current time
    current_time = time.strftime('%H:%M:%S')
    # Update the clock label text
    clock.config(text=current_time)
    # Schedule the next update after 1 second (1000 milliseconds)
    clock.after(1000, tick)

# Function to handle GUI setup
def setup_gui():
    global id_entry, name_entry, total_present, graph_label, clock, tv
    total_present = 0  # Initialize total_present count

    global key
    key = ''
    
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    day, month, year = date.split("-")

    mont = {'01': 'January',
            '02': 'February',
            '03': 'March',
            '04': 'April',
            '05': 'May',
            '06': 'June',
            '07': 'July',
            '08': 'August',
            '09': 'September',
            '10': 'October',
            '11': 'November',
            '12': 'December'
            }

    window = tk.Tk()
    window.geometry("1280x720")
    window.resizable(True, False)
    window.title("Attendance Management System Using Face Recognition")

    # Frame for login and registration section
    frame1 = tk.Frame(window, bg="#595959")
    frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.7)

    # Frame for attendance report and plot graph
    frame2 = tk.Frame(window, bg="#595959")
    frame2.place(relx=0.51, rely=0.195, relwidth=0.48, relheight=0.65)

    message3 = tk.Label(window, text="CSCE 5214: Software Development for AI ", font=('centaur', 29, ' bold '))
    message3.place(relx=0.5, rely=0.01, anchor='n')

    frame3 = tk.Frame(window, bg="#efefef")
    frame3.place(relx=0.26, rely=0.09, relwidth=0.44, relheight=0.04)

    datetime_label = tk.Label(frame3, text=f"{day} {mont[month]} {year} | ", font=('centaur', 20, ' bold '))
    datetime_label.pack(side="left", padx=(141, 0))

    # Create a label for displaying the clock
    clock = tk.Label(frame3, font=('centaur', 20, ' bold '))
    clock.pack(side="left", padx=(0, 0))
    # Start updating the clock
    tick()

    # Enrollment Section
    head1 = tk.Label(frame1, text="       Student's Login and Registration Section                       ", fg="#ffffff",
                     bg="#00853E", font=('centaur', 18, ' bold '))
    head1.grid(row=0, column=0)

    id_label = tk.Label(frame1, text="Enter UNT ID", fg="#ffffff", bg="#595959", font=('centaur', 15, ' bold '))
    id_label.place(x=10, y=40)
    id_entry = tk.Entry(frame1, width=18, fg="black", font=('centaur', 15, ' bold '))
    id_entry.place(x=225, y=40)

    name_label = tk.Label(frame1, text="Enter Student Name", fg="#ffffff", bg="#595959",
                          font=('centaur', 15, ' bold '))
    name_label.place(x=10, y=80)
    name_entry = tk.Entry(frame1, width=18, fg="black", font=('centaur', 15, ' bold '))
    name_entry.place(x=225, y=80)

    register_button = tk.Button(frame1, text="Register", command=register_new_user, fg="white", bg="#00853E",
                                activebackground="white", font=('centaur', 12, ' bold '))
    register_button.place(x=100, y=140)

    login_button = tk.Button(frame1, text="Login", command=login, fg="white", bg="#00853E", activebackground="white",
                             font=('centaur', 12, ' bold '))
    login_button.place(x=260, y=140)

    # Attendance Summary
    head2 = tk.Label(frame2, text=" Summary of the Student's Attendance                       ", fg="#ffffff",
                     bg="#00853E", font=('centaur', 18, ' bold '))
    head2.place(x=0, y=0)

    # Create Treeview widget for attendance report
    tv = ttk.Treeview(frame1, columns=('Student ID', 'Student Name', 'Login Time'), show='headings', height=10)
    tv.column('#0',width=82)
    tv.column('Student ID',width=130)
    tv.column('Student Name',width=136)
    tv.column('Login Time',width=133)
    tv.grid(row=2,column=0,padx=(0,150),pady=(150,0),columnspan=4)
    tv.heading('Student ID', text='Student ID')
    tv.heading('Student Name', text='Student Name')
    tv.heading('Login Time', text='Login Time')
    #tv.pack(fill='both', expand=True)

    # Label to display the plot graph
    graph_label = tk.Label(frame2, bg="#595959")
    graph_label.pack(fill='both', expand=True)


    window.mainloop()

# Multithreaded version of GUI setup
def setup_gui_threaded():
    threading.Thread(target=setup_gui).start()

# Run GUI setup in a separate thread
setup_gui_threaded()
