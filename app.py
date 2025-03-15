import streamlit as st
import pymysql
import smtplib
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO(r"C:\Users\subin\OneDrive\Desktop\Deep Learning\Helmet Detection\runs\detect\train\weights\best.pt")  

# Function to connect to MySQL
def connect_to_db():
    try:
        return pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            database=os.getenv("DB_NAME"),
            autocommit=True
        )
    except pymysql.MySQLError as e:
        st.error(f"❌ Database Connection Error: {e}")
        return None

# Function to send email
def send_email(email, plate):
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
        
        subject = "Helmet Violation Alert "
        message = f"Dear User,\n\nOur system detected that you were riding without a helmet.\n\nLicense Plate: {plate}\n\nFine: 1000\n\nPlease wear a helmet for safety!\n\nRegards,\nTraffic Safety Team"

        server.sendmail(os.getenv("EMAIL_USER"), email, f"Subject: {subject}\n\n{message}")
        server.quit()
        st.success(f" Email sent to {email}")
    
    except Exception as e:
        st.error(f" Failed to send email: {e}")

# Helmet detection function
def detect_helmet(image):
    results = model.predict(image, save=True, conf=0.5)
    
    detected_motorcyclists = []
    detected_helmets = []
    detected_license_plates = []
    
    for r in results:
        for i, box in enumerate(r.boxes.xyxy):
            label = r.names[int(r.boxes.cls[i])].lower()
            x1, y1, x2, y2 = map(int, box)
            
            if label == "motorcyclist":
                detected_motorcyclists.append((x1, y1, x2, y2))
            elif label == "helmet":
                detected_helmets.append((x1, y1, x2, y2))
            elif label == "license_plate":
                detected_license_plates.append((x1, y1, x2, y2))
    
    return detected_motorcyclists, detected_helmets, detected_license_plates
def extract_license_plate_text(plate_coords,image):
    st.write("Entered OCR function")
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = plate_coords
    plate_img = image[y1:y2, x1:x2]

    if plate_img is None or plate_img.size == 0:
       st.write("Error: Extracted plate image is empty!")
       return None

    # Convert to grayscale
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing for better OCR
    plate_img = cv2.GaussianBlur(plate_img, (5, 5), 0)  # Reduce noise
    plate_img = cv2.adaptiveThreshold(plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Debugging: Save and Display the Image
    # cv2.imwrite("debug_plate.jpg", plate_img)
    # plt.imshow(plate_img,cmap='gray')
    # plt.axis("off")
    # plt.show()

   
    # Crop the upper half
    plate_upper = plate_img[7:plate_img.shape[0] // 2+1, 1:-2]
    

    # Crop the lower half
    plate_lower = plate_img[plate_img.shape[0] // 2-4:-1, 3:]
    
    # OCR with character whitelist
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.'
    
    # Extract text separately
    upper_text = pytesseract.image_to_string(plate_upper, config=custom_config).strip()
    lower_text = pytesseract.image_to_string(plate_lower, config=custom_config).strip()

    # Combine results
    license_plate_text = upper_text +" "+lower_text


    st.write(f'Extracted Text:{license_plate_text}')

    return license_plate_text

# Streamlit UI
st.title("🛵 Helmet Detection System")
st.write("Upload an image to check for helmet violations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 detection
    detected_motorcyclists, detected_helmets, detected_license_plates = detect_helmet(image_bgr)

    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Processing results
    # Process detected motorcyclists
    for mc in detected_motorcyclists:
        mc_x1, mc_y1, mc_x2, mc_y2 = mc
        helmet_found = False
        license_plate_text = None

        # Check if helmet is present
        for h in detected_helmets:
            h_x1, h_y1, h_x2, h_y2 = h
            if h_x1 > mc_x1 and h_x2 < mc_x2 and h_y1 > mc_y1 and h_y2 < mc_y2:
                helmet_found = True
                break

        # Find corresponding license plate
        for lp in detected_license_plates:
            lp_x1, lp_y1, lp_x2, lp_y2 = lp
            
            if mc_x1 < lp_x1 and lp_x2 < mc_x2 and mc_y1 < lp_y1 and lp_y2 < mc_y2:
                license_plate_text = extract_license_plate_text(lp,image_bgr)
                break

        
        if not helmet_found and license_plate_text:
            st.warning(f"⚠ Violation Detected! License Plate: {license_plate_text}")

            # Save to Database
            conn = connect_to_db()
            if conn:
                st.success("✅ Connected to MySQL!")
                cursor = conn.cursor()
                query = "INSERT INTO detected_plates (license_plate, detection_time) VALUES (%s, NOW())"
                cursor.execute(query, (license_plate_text,))
                conn.commit()
                conn.close()
                st.success(f"✅ License Plate {license_plate_text} stored in the database!")

            # Fetch Email from Database
            conn = connect_to_db()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT email FROM vehicles_details WHERE licence_plate = %s", (license_plate_text,))
                result = cursor.fetchone()
                conn.close()

                if result:
                    user_email = result[0]
                    st.success(f"✅ Found user! Sending email to {user_email}...")
                    send_email(user_email, license_plate_text)
                else:
                    st.error("⚠ No email found for this license plate.")
                
            else:
                st.error("❌ Could not connect to the database. Check your MySQL credentials.")
