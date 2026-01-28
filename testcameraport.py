import cv2

# Initialize the camera at index 0
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera 0.")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame.")
            break

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Press 'q' on the keyboard to exit
        if cv2.waitKey(1) == ord('q'):
            break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
