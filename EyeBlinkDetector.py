from flask import Flask, render_template, request, redirect, url_for
import time
import cv2
import cvzone
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from FaceDetection import FaceDectector
from cvzone.PlotModule import LivePlot

app = Flask(__name__)

cap = cv2.VideoCapture(0)
pTime = 0
fd = FaceDectector(maxFaces=1)
left_eye_point = [22, 23, 24, 26, 110, 157, 158, 159, 160, 130, 243]
plotGraph = LivePlot(640, 480, [20, 50])
ratioList = []
blinkCounter = 0
counter = 0

# Create Attendance folder if not exists
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_detection():
    global blinkCounter, counter, ratioList
    while True:
        success, img = cap.read()
        img, faces = fd.detectFace(img, draw=False)

        if faces:
            face = faces[0][1]
            for id in left_eye_point:
                cv2.circle(img, face[id][1:], 3, (255, 0, 255), cv2.FILLED)

            leftUp = face[159][1:]
            leftDown = face[23][1:]
            leftLeft = face[130][1:]
            leftRight = face[243][1:]
            length_vertical, info, img = fd.findDistance(leftUp, leftDown, img)
            length_horizontal, info2, img = fd.findDistance(leftLeft, leftRight, img)
            ratio = (length_vertical / length_horizontal) * 100
            ratioList.append(ratio)

            if len(ratioList) > 10:
                ratioList.pop(0)
            ratioAvg = sum(ratioList) / len(ratioList)

            if ratioAvg < 29 and counter == 0:
                blinkCounter += 1
                counter = 1
            if counter != 0:
                counter += 1
                if counter > 15:
                    counter = 0

            cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', [20, 450], 1, 1)
            imgPlot = plotGraph.update(ratioAvg)
            joinedImg = cvzone.stackImages([img, imgPlot], 2, 1)
        else:
            joinedImg = cvzone.stackImages([img, img], 2, 1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)

        if blinkCounter >= 2:
            name = f'Student_{blinkCounter}'
            usn = f'1MJ23CS{blinkCounter:03d}'
            filename = f'Attendance/{name}_{usn}.txt'

            with open(filename, 'w') as f:
                f.write(f'Name: {name}\n')
                f.write(f'USN: {usn}\n')
                f.write('Attendance Marked\n')

            cv2.imwrite(f'Attendance/{name}_{usn}.jpg', img)

            # Store the name and USN to use later when sending the email
            return redirect(url_for('email_form', name=name, usn=usn, filename=filename, img_path=f'Attendance/{name}_{usn}.jpg'))

        cv2.putText(img, "FPS:" + str(int(fps)), (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.imshow("image", joinedImg)
        cv2.waitKey(1)

    return redirect(url_for('index'))

@app.route('/email_form')
def email_form():
    name = request.args.get('name')
    usn = request.args.get('usn')
    filename = request.args.get('filename')
    img_path = request.args.get('img_path')
    return render_template('email_form.html', name=name, usn=usn, filename=filename, img_path=img_path)

@app.route('/submit_email', methods=['POST'])
def submit_email():
    user_email = request.form['email']
    name = request.form['name']
    usn = request.form['usn']
    filename = request.form['filename']
    img_path = request.form['img_path']

    # Default email
    default_email = 'defauilt recever email'

    # Send the email with the attached files
    send_email(user_email, filename, img_path)
    send_email(default_email, filename, img_path)

    return redirect(url_for('index'))

def send_email(user_email, txt_file, img_file):
    msg = MIMEMultipart()
    msg['From'] = 'sender email'
    msg['To'] = user_email
    msg['Subject'] = 'Attendance Record'

    body = 'Attached is the attendance record and image.'
    msg.attach(MIMEText(body, 'plain'))

    # Attach text file
    with open(txt_file, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(txt_file)}')
        msg.attach(part)

    # Attach image file
    with open(img_file, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(img_file)}')
        msg.attach(part)

    # Email configuration
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
    SENDER_EMAIL = 'sender email'
    SENDER_PASSWORD = 'sender app password '

    # Send email
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)
    server.quit()

if __name__ == '__main__':
    app.run(debug=True)
