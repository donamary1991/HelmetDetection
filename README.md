# HelmetDetection
Helmet Detection using YOLOv8 :
This project detects whether a motorcyclist is wearing a helmet using YOLOv8. If a helmet is missing, an email notification is sent using details from an SQL database.

Features

✔ Helmet detection using YOLOv8

✔ Object detection for helmets, license plates, and motorcyclists

✔ Email notification if a helmet is not worn

✔ MySQL database integration for user details

✔ Works with real-time video streams & images



How the App Works:
 
Upload an image.

YOLOv8 detects motorcyclists, helmets, and license plates.

If a motorcyclist is not wearing a helmet, their license plate is extracted.

The license plate is stored in the MySQL database.

The app fetches the registered email of the vehicle owner from the database.

An email is sent to the owner about the violation.
