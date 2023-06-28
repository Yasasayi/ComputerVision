from PyQt5.QtWidgets import *
import sys
import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Rock Scissors Paper ')
        self.setGeometry(200, 200, 300, 200)
        self.label = QLabel('', self)
        
        startButton = QPushButton('start', self)
        startButton.setGeometry(100, 30, 100, 50)
        startButton.clicked.connect(self.rsp)
        self.label.setGeometry(90, 50, 200, 100)
        
        self.show()
    
    def rsp(self):
        enemy = self.enemy_rsp()
        player = self.player_rsp()
        
        flag = 0 # 0 = 무승부, 1 = 컴퓨터 승, 2: 플레이어 승
        result = 'Draw'
        
        if enemy == '가위':
            if player == '보':
                flag = 1
            elif player == '바위':
                flag = 2
        elif enemy == '바위':
            if player == '가위':
                flag = 1
            elif player == '보':
                flag = 2
        elif enemy == '보':
            if player == '바위':
                flag = 1
            elif player == '가위':
                flag = 2
        if flag == 1:
            result = 'You Lose..'
        elif flag == 2:
            result = 'You Win!'
        self.label.setText("CPU :" + enemy + "\tYou: " + player + "\n" + result)
        
    def enemy_rsp(self):
        random = np.random.randint(3, size=1)
        flag = ''
        if random == 0:
            flag = '가위'
        elif random == 1:
            flag = '바위'
        else:
            flag = '보'
        return flag
                
    def player_rsp(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        mp_drawing_styles = mp.solutions.drawing_styles

        cap = cv.VideoCapture(0)

        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        while cap.isOpened():
            success, image = cap.read()

            if not success:
                print('Ignoring empty camera frame.')
                continue

            image = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = hands.process(image)


            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:


                    # 손가락을 일자로 편 상태인지 확인
                    thumb_finger_state = 0
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height:
                            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height:
                                thumb_finger_state = 1

                    index_finger_state = 0
                    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height:
                            if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height:
                                    index_finger_state = 1

                    middle_finger_state = 0
                    if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height:
                            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height:
                                middle_finger_state = 1

                    ring_finger_state = 0
                    if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height:
                            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height:
                                ring_finger_state = 1

                    pinky_finger_state = 0
                    if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height:
                        if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height:
                            if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height:
                                pinky_finger_state = 1

                font = ImageFont.truetype('fonts/gulim.ttc', 80)
                image = Image.fromarray(image)
                draw = ImageDraw.Draw(image)

                    
                flag = ''
                if thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 1 and pinky_finger_state == 1:
                    flag = '보'
                elif thumb_finger_state == 1 and index_finger_state == 1 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    flag = '가위'
                elif thumb_finger_state == 0 and index_finger_state == 1 and middle_finger_state == 1 and ring_finger_state == 0 and pinky_finger_state == 0:
                    flag = '가위'
                elif thumb_finger_state == 0 and index_finger_state == 0 and middle_finger_state == 0 and ring_finger_state == 0 and pinky_finger_state == 0:
                    flag = '바위'

                w, h = font.getsize(flag)

                x = 50
                y = 50

                draw.rectangle((x, y, x + w, y + h), fill='black')
                draw.text((x, y),  flag, font=font, fill=(255, 255, 255))
                image = np.array(image)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            cv.imshow('Hands', image)

            if cv.waitKey(5) == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()
        return flag

app = QApplication(sys.argv)
win = MainWindow()
app.exec_()