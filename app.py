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
model = YOLO(r"C:\Users\subin\OneDrive\Desktop\Deep Learning\Helmet_Detection_Violation_Alert\runs\detect\train6\weights\best.pt")  

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
    # # Example path to your predicted image
    for r in results:
        dir=r.save_dir
        img=os.listdir(dir)[0]
        image_path=os.path.join(dir,img)
        img_read=cv2.imread(image_path)
        img_rgb=cv2.cvtColor(img_read,cv2.COLOR_BGR2RGB)
        # Display image
        st.image(img_rgb, caption="Predicted Image", use_container_width=True)
        
        

    
    detected_motorcyclists = []
    detected_nohelmets = []
    detected_number_plates = []
    
    for r in results:
        for i, box in enumerate(r.boxes.xyxy):
            label = r.names[int(r.boxes.cls[i])].lower()
            x1, y1, x2, y2 = map(int, box)
            
            if label == "rider":
                detected_motorcyclists.append((x1, y1, x2, y2))
            elif label == "without helmet":
                detected_nohelmets.append((x1, y1, x2, y2))
            elif label == "number plate":
                detected_number_plates.append((x1, y1, x2, y2))
    
    return detected_motorcyclists, detected_nohelmets, detected_number_plates
def extract_license_plate_text(plate_coords,image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = plate_coords
    plate_img = image[y1:y2, x1:x2]

    if plate_img is None or plate_img.size == 0:
       st.write("Error: Extracted plate image is empty!")
       return None

    # Convert to grayscale
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)

    # Apply preprocessing for better OCR
    plate_img = cv2.GaussianBlur(plate_img, (5, 5), 0)  # Reduce noise
    plate_img = cv2.bilateralFilter(plate_img, 12, 17, 17)  # Smoothing but preserving edges
    plate_img = cv2.adaptiveThreshold(plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plate_img = cv2.resize(plate_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # # Debugging: Save and Display the Image
    # cv2.imwrite("number_plate.jpg", plate_img)
    # plt.imshow(plate_img,cmap='gray')
    # plt.axis("off")
    # plt.show()

   
    
    # OCR with character whitelist
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLLMNOPQRSTUVWXYZ0123456789.'


    
    # Extract text separately
    numberplate_text = pytesseract.image_to_string(plate_img,config= custom_config).strip()
    
    


    st.write(f'Extracted Text:{numberplate_text}')

    return numberplate_text

# Streamlit UI
st.title("🛵 Helmet Detection System")
st.write("Upload an image to check for helmet violations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # Display image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 detection
    detected_motorcyclists, detected_wohelmets, detected_number_plates = detect_helmet(image_bgr)
    

    # Processing results
    # Process detected motorcyclists
    for woh in detected_wohelmets:
        wx1,wy1,wx2,wy2=woh
        helmet_found=False
        for rider in detected_motorcyclists:
            rx1,ry1,rx2,ry2=rider
            if rx1<wx1 and ry1<wy1 and rx2>wx2 and ry2>wy2:
                for np in detected_number_plates:
                    np_x1, np_y1, np_x2, np_y2 = np
                    if rx1 < np_x1 and np_x2 < rx2 and ry1 < np_y1 and np_y2 < ry2:
                        license_plate_text = extract_license_plate_text(np,image_bgr)
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
