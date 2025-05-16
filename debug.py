import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
print("Frame read:", ret)
cap.release()
