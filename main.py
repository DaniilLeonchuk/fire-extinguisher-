import re
import cv2
import pytesseract
import easyocr
import numpy as np


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class Fire_exi_tag():
    

    def __init__(self, img):
        self.image = cv2.imread(img)
        if self.image is None:
            raise ValueError(f'Не загрузилось изображение {img}')
        

    def proccesing_img(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(gray, 255, 
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 31, 15)

        kenary = np.ones((1,1), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kenary, iterations=1)
        processed = cv2.dilate(processed, kenary, iterations=1)
        processed = cv2.bitwise_not(processed)


        return processed
        
    






if __name__ == '__main__':
    tag = Fire_exi_tag('fire_exti/photo16.jpg')
    




    processed_img = tag.proccesing_img()
    cv2.imshow('tag', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
