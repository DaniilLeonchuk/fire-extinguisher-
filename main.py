import re
import cv2
from PIL import ImageFont, ImageDraw, Image
import easyocr
import numpy as np
from datetime import datetime





class Fire_exi_tag():
    

    def __init__(self, img):
        self.image = cv2.imread(img)
        if self.image is None:
            raise ValueError(f'Не загрузилось изображение {img}')
    


        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)

        try:
            self.font = ImageFont.truetype("arial.ttf", 24)  # или другой подходящий шрифт
        except IOError:
            print("Ошибка: шрифт 'arial.ttf' не найден.")
            self.font = None
        

  
    def proccesing_img(self, gamma=1.5, clip_limit=2.0):
        
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
        
        # Мягкое увеличение резкости
        kernel = np.array([[0, -0.25, 0],
                   [-0.25, 2, -0.25],
                   [0, -0.25, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # CLAHE с настраиваемым пределом
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(sharpened)
        return enhanced 
        


    


    #Обработка изображения (доработать)
    



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
    
    def correct_ocr_text(self, text):
        """Постобработка текста (метод 4)"""
        corrections = {
            '0П': 'ОП', 
            'Г0': 'ГО',
            'дaтa': 'дата',
            'П0ЖАР': 'ПОЖАР'
        }
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        return re.sub(r'[^а-яА-Я0-9.,\-: ]', '', text).strip()
    

    

#на этой эмблеме ищет где написан год и месяц последующей поверки, 
#распознает что там написано и в выдает в качестве результата своей работы.

    def detect_word_date(self, keywords=None):
        if keywords is None:
            keywords = {
                'дата заправки': 'Дата заправки',
                'дата изготовления': 'Дата изготовления',
                'дата следующего освидетельствования': 'Дата следующего освидетельствования',
                'год': 'Год изготовления',
                'дата': 'дата'
            }

        text = self.easyocr_img()
        if isinstance(text, bytes):
            text = text.decode('utf-8')

        found_dates = []
        used_positions = set()

        # Улучшенные шаблоны для текстовых месяцев
        month_pattern = r'(январ[ья]|феврал[ья]|март[а]?|апрел[ья]|ма[йя]|июн[ья]|июл[ья]|август[а]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])'
        
        # Шаблоны с ключевыми словами
        for keyword, label in keywords.items():
            pattern = fr'{re.escape(keyword)}\s*[:\-]?\s*(\d{{2}}\.\d{{2}}\.\d{{4}}|\d{{2}}\.\d{{4}}|\d{{2}}/\d{{4}}|\d{{2}}-\d{{4}}|{month_pattern}\s\d{{4}})'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()
                if (start, end) not in used_positions:
                    used_positions.add((start, end))
                    date = match.group(1)
                    found_dates.append({
                        'type': label,
                        'date': date,
                        'formatted_date': date.title(),  # Первая буква заглавная
                        'raw': match.group(0)
                    })

        # Шаблоны для дат без ключевых слов
        general_patterns = [
            fr'{month_pattern}\s(?:19\d{{2}}|20[0-8]\d|209[0-9])',  # Текстовый месяц + год (до 2099)
            r'\d{2}\.\d{2}\.(?:19\d{2}|20[0-8]\d|209[0-9])',        # ДД.ММ.ГГГГ (до 2099)
            r'\d{2}\.(?:19\d{2}|20[0-8]\d|209[0-9])',               # ММ.ГГГГ (до 2099)
            r'(?:19\d{2}|20[0-8]\d|209[0-9])'                        # ГГГГ
        ]

        for pattern in general_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start, end = match.span()
                if not any(start >= used_start and end <= used_end for used_start, used_end in used_positions):
                    used_positions.add((start, end))
                    date = match.group()
                    found_dates.append({
                        'type': 'Дата',
                        'date': date,
                        'formatted_date': date.title(),
                        'raw': match.group()
                    })

        return found_dates if found_dates else [{'type': 'Дата не найдена', 'date': None, 'formatted_date': None, 'raw': None}]
        
    

        



        
        

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
    tag = Fire_exi_tag('fire_exti/photo9.jpg')
    
    

    #Вывод даты
    dates = tag.detect_word_date()
    print('Найденные даты:')
    for i, date_info in enumerate(dates, 1):
        if date_info['type'] == 'Дата не найдена':
            print(date_info['type'])
        else:
            print(f'{i}. {date_info['type']}: {date_info['date']}')
        

    

    
    
    #Вывод цвета огнетушителя
    color_fire = tag.detect_color()
    print(f'Цвет огнетушителя: {color_fire}')

    print()
    #Вывод текста с помощью easyocr
    text = tag.easyocr_img()
    print("Распознанный текст:", tag.easyocr_img())
    
    #Вывод изображения для контроля обработки качества
    proccesing_img = tag.proccesing_img()
 
    cv2.imshow('tag', proccesing_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
