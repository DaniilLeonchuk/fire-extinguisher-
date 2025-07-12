import re
import cv2
from PIL import ImageFont, ImageDraw, Image
import easyocr
import numpy as np
from datetime import datetime
import skimage as ski






class Fire_exi_tag():
    

    def __init__(self, img):
        self.image = cv2.imread(img)
        if self.image is None:
            raise ValueError(f'Не загрузилось изображение {img}')
    


        self.reader = easyocr.Reader(['ru', 'en'], gpu=False, )


    

  

        


    #Обработка изображения (доработать)
    def proccesing_img(self):

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        return clahe.apply(gray)

 #Все остальное, что распознано тоже выдает в строку, как доп. информация (добавить )
    


    #Чтение текста с помощью easyocr
    def easyocr_img(self):

        
        img = self.proccesing_img()
        
        results = self.reader.readtext(img, 
                                       batch_size=1, 
                                       detail=1, 
                                       decoder='beamsearch',
                                       
        )
        
       
        
        
        return " ".join([result[1] for result in results]) if results else "" 
    #Добавить новые библиотеки для распознавания текс на изображении
        


#на этой эмблеме ищет где написан год и месяц последующей поверки, 
#распознает что там написано и в выдает в качестве результата своей работы.

    def detect_word_date(self, keywords=None):

       
        if keywords is None:
            keywords = [
        'поверка', 'проверка', 'следующая', 'дата', 
        'годен до', 'срок', 'испытания', 'испытан',
        'ОТК', 'ПОВЕРКА', 'ПОВЕРЕНО', 'ПРОВЕРЕНО',
        'следующая проверка', 'дата поверки',
        'год', 'ГОД', 'Год', 'Дата заправки'
        ]

        text = self.easyocr_img()
        if isinstance(text, bytes):
            text = text.decode('utf-8')

        text_lower = text.lower()
    
     
        date_patterns = [
            r'(поверк[аи]|проверк[аи]|годен до|срок|испытан до)[:\s]*(\d{2}\.\d{2}\.\d{4})',
            r'(следующая проверка|дата поверки)[:\s]*(\d{2}\.\d{4})',
            

            r'\b\d{2}\.\d{2}\.\d{4}\b',     
            r'\b\d{2}\.\d{4}\b',             
            r'\b\d{2}/\d{4}\b',              
            r'\b\d{2}-\d{4}\b',              
            r'\b(?:январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья]) \d{4}\b'

        ]


        found_dates = []
        

        for keyword in keywords:
            for pattern in date_patterns:
                regex_pattern = fr'{re.escape(keyword)}\s*[:\-]?\s*({pattern})'
                
                matches = re.finditer(regex_pattern, text, re.IGNORECASE)
                for match in matches:
                    date = match.group(1)
                    if date not in found_dates:
                        found_dates.append(date) 

        # Если не нашли по ключевым словам, ищем любую дату в тексте
        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                date = match.group()
                if date not in found_dates:
                    found_dates.append(date)

        return found_dates if found_dates else 'Дата не найдена'
        
    

        



        
        

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
        

        if colors[max_color] < (hsv.shape[0] * hsv.shape[1] * 0.3):  
            return "Не удалось определить цвет огнетушителя"
    
        return max_color






if __name__ == '__main__':
    tag = Fire_exi_tag('photo.png')
    
    

    #Вывод даты
    dates = tag.detect_word_date()
    print('Найденные даты:')
    for i, date in enumerate(dates, 1):
        print(f'{i}. {date}')

    

    
    print()
    #Вывод цвета огнетушителя
    color_fire = tag.detect_color()
    print(f'Цвет огнетушителя: {color_fire}')

    
    #Вывод текста с помощью easyocr
    text_easyocr = tag.easyocr_img()
    print(f'Дополнительная информация: \n{text_easyocr}')
    
    #Вывод изображения для контроля обработки качества
    proccesing_img = tag.proccesing_img()
 
    cv2.imshow('tag', proccesing_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
