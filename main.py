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
        
    #Обработка изображения
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

 #Все остальное, что распознано тоже выдает в строку, как доп. информация (добавить )
    #Чтение текста с помощью pytesseract
    def pytesseract_img(self):

        img = self.proccesing_img()

        config = '--oem 3 --psm 6 -l rus'
        tex_pytes =  pytesseract.image_to_string(img, config=config)

        return tex_pytes


    #Чтение текста с помощью easyocr
    def easyocr_img(self):

        img = self.proccesing_img()

        reader = easyocr.Reader(['ru'], gpu=False)

        result = reader.readtext(img)

        texts = [detect[1] for detect in result]
        return texts[0] if texts else ""

    #Добавить новые библиотеки для распознавания текс на изображении
        


#на этой эмблеме ищет где написан год и месяц последующей поверки, 
#распознает что там написано и в выдает в качестве результата своей работы.

    def detect_word_date(self):
        pass

#Проверяет, чтобы цвет фона был цвет огнетушителя: красный, оранжевый И так далее

    def detect_color(self):
        
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)


        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 120, 70])
        red_upper2 = np.array([180, 255, 255])
    
        yellow_lower = np.array([15, 120, 70])
        yellow_upper = np.array([35, 255, 255])

        orange_lower = np.array([8, 150, 100])
        orange_upper = np.array([15, 255, 255])
        
        green_lower = np.array([36, 120, 70])
        green_upper = np.array([85, 255, 255])
    
        blue_lower = np.array([86, 120, 70])
        blue_upper = np.array([130, 255, 255])

        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)


        colors = {
        'Красный': cv2.countNonZero(red_mask),
        'Оранжевый': cv2.countNonZero(orange_mask),
        'Желтый': cv2.countNonZero(yellow_mask),
        'Зеленый': cv2.countNonZero(green_mask),
        'Синий': cv2.countNonZero(blue_mask),
        }


        max_color = max(colors, key=colors.get)
        

        if colors[max_color] < (hsv.shape[0] * hsv.shape[1] * 0.1):  
            return "Не удалось определить цвет огнетушителя"
    
        return max_color






if __name__ == '__main__':
    tag = Fire_exi_tag('color_to_check/black.jpg')
    
    


    #Вывод текст с помощью pytesseract
    text_pysseract = tag.pytesseract_img()
    print(f'Расшифрованный текст: \n{text_pysseract}')


    #Вывод текста с помощью easyocr
    text_easyocr = tag.easyocr_img()
    print(f'Расшифрованный текст: \n{text_easyocr}')


    

    #Вывод цвета огнетушителя
    color_fire = tag.detect_color()
    print(f'Цвет огнетушителя: {color_fire}')




    #Вывод изображения для контроля обработки качества
    processed_img = tag.proccesing_img()
    cv2.imshow('tag', processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
