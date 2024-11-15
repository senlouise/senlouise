import cv2
import pytesseract

# ติดตั้ง Tesseract OCR Path (ตั้งค่าเฉพาะเครื่อง)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_license_plate(image_path):
    # อ่านรูปภาพ
    image = cv2.imread(image_path)

    # แปลงเป็นสีเทาเพื่อเพิ่มประสิทธิภาพการตรวจจับ
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ใช้การตรวจจับขอบ
    edged_image = cv2.Canny(gray_image, 30, 200)

    # ค้นหาขอบเขตของทะเบียนรถ
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    license_plate = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:  # สี่เหลี่ยมผืนผ้าคือรูปทะเบียนรถ
            license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            license_roi = gray_image[y:y + h, x:x + w]
            break

    if license_plate is not None:
        # ใช้ OCR เพื่ออ่านข้อความในทะเบียน
        text = pytesseract.image_to_string(license_roi, config='--psm 8')
        return text.strip()
    else:
        return "License plate not found"

# ทดสอบโปรแกรม
image_path = 'car_image.jpg'  # ระบุ path รูปภาพ
license_number = detect_license_plate(image_path)
print(f"Detected License Plate: {license_number}")
