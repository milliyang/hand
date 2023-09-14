from math import hypot

import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

from hand_utils import detectHands, formatLandmarks, drawLine, addText

"""
    Control the brightness of the primary monitor via your fingers
"""

# Order: Blue, Green, Red
DOT_COLOR = (0, 0, 255)
CONNECTION_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 0)

EXIT_KEY = 'q'

BRIGHTNESS_MIN = 0
BRIGHTNESS_MAX = 100

# Don't change
THUMB_TIP = 4
MIDDLE_TIP = 12

def main():
    mpDrawing = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands

    cap = cv2.VideoCapture(0)

    with mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:

        while (cap.isOpened()):
            success, frame = cap.read()

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            image, results = detectHands(hands, frame)

            lm = []
            landmarks = results.multi_hand_landmarks
            if (landmarks):
                lm = formatLandmarks(landmarks, image, mpDrawing, mpHands, DOT_COLOR, CONNECTION_COLOR)

            if (lm != []):
                # Calculates the brightness with finger's distance

                x1, y1 = lm[THUMB_TIP]['cx'], lm[THUMB_TIP]['cy']
                x2, y2 = lm[MIDDLE_TIP]['cx'], lm[MIDDLE_TIP]['cy']

                distance = hypot(x2 - x1, y2 - y1)
                b = np.interp(
                    distance,
                    [15, 220],
                    [BRIGHTNESS_MIN, BRIGHTNESS_MAX]
                )
                b = int(b)

                drawLine(image, lm, THUMB_TIP, MIDDLE_TIP, LINE_COLOR)
                addText(image, lm, f'Brightness: {b}', THUMB_TIP, MIDDLE_TIP, TEXT_COLOR)

                print(f'Set brightness to: {b}')
                sbc.set_brightness(b)

            cv2.imshow('Primary monitor brightness control by hand', image)

            # Exit if user pressed 'q'
            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break
            
    cap.release()
    cv2.destroyAllWindows()

    print('--- End of Program ---')

if (__name__ == '__main__'):
    main()