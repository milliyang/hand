from math import hypot

import cv2
import mediapipe as mp
import numpy as np

from hand_utils import getAudioDevice, detectHands, formatLandmarks, drawLine, addText

"""
    Volume control by hand (Tip of thumb + index finger)
"""

# https://google.github.io/mediapipe/images/mobile/hand_landmarks.png

# Order: Blue, Green, Red
DOT_COLOR = (0, 0, 255)
CONNECTION_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 255)
TEXT_COLOR = (255, 255, 0)

EXIT_KEY = 'q'

# Don't change
THUMB_TIP = 4
INDEX_TIP = 8

def main():
    mpDrawing = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands

    volume = getAudioDevice()
    volMin, volMax = volume.GetVolumeRange()[::2]

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
                vol = setVolume(lm, volume, volMin, volMax)
                drawLine(image, lm, THUMB_TIP, INDEX_TIP, LINE_COLOR)
                
                text = f'Volume: {vol:.2f}'
                addText(image, lm, text, THUMB_TIP, INDEX_TIP, TEXT_COLOR)

            cv2.imshow('Volume control by hand', image)

            # Exit if user pressed 'q'
            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("--- End of Program ---")

def setVolume(landmarks, volume, volMin, volMax):
    """
        Set system volume and return volume from calculation
    """
    
    x1, y1 = landmarks[THUMB_TIP]['cx'], landmarks[THUMB_TIP]['cy']
    x2, y2 = landmarks[INDEX_TIP]['cx'], landmarks[INDEX_TIP]['cy']


    distance = hypot(x2 - x1, y2 - y1)
    v = np.interp(
        distance,
        [15, 220],
        [volMin, volMax]
    )

    """
    # This we don't check the value of v, it will cause error when setting the volume
    if ((v <= volMin) or (v >= volMax)):
        print('[WARNING] Boundary reached! Please adjust distance between the hand and camera!')
        return
    """

    print(f'volMin = {volMin} | volMax = {volMax} | v = {v}')
    try:
        volume.SetMasterVolumeLevel(v, None)
    except:
        # pass
        print('[WARNING] Boundary reached! Please adjust distance between the hand and camera!')
    
    return v

if (__name__ == '__main__'):
    main()