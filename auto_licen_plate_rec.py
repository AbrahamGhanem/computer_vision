import cv2
import pytesseract
import numpy as np
import os


# 1. install pytesseract pip and exe and then add to env path
# 2. import an img -> resize it if it is small -> apply binary threshold -> pytesseract.image_to_string(threshold)
# 3. if images are skewed there should be some pre-processing to align before recognition

# we need to specify the installation path where the exe is
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
classifier_pth = r'C:\Users\GHANEM\anaconda3\envs\facebook-detectron\Lib\site-packages\cv2\data\haarcascade_russian_plate_number.xml'
cascade = cv2.CascadeClassifier(classifier_pth)
states = {'DO': 'Stadt Dortmund',
          'BO': 'Stadt Bochum'
          }


def extract_num(image_name, root):
    global read
    img_path = os.path.join(root, image_name)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)  # img, scaleFactor, MinNeighbours
    for (x, y, w, h) in nplate:
        # crop the number plate
        a, b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b, :]

        # img processing
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)  #  compute the max pixel value overlapped by the kernel
        plate = cv2.erode(plate, kernel, iterations=3)  #  compute the min pixel value overlapped by the kernel
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 115, 255, 0, cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(letter for letter in read if letter.isalnum())
        state_sym = read[0:2]

        try:
            print('Dieses Auto gehört zu: ', states[state_sym])
        except:
            print('Unbekannt!')

        print(read)
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 255, 255), thickness=3)
        cv2.imshow('img_with_bbox', plate)

    cv2.imshow('result', img)
    cv2.imwrite('result.jpg', img)
    cv2.waitKey(0)


root = r'C:\Users\GHANEM\Desktop\OpenCV\data\car_plates'
image_name = 'car_plates_2.jpg'
extract_num(image_name, root)
