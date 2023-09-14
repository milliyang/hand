import cv2
import mediapipe as mp
import win32api, win32con

from math import hypot

from hand_utils import detectHands, formatLandmarks

"""
    14/11/2021
    - Use index finger's tip to move the cursor
    - Perform click / drag by touch the middle part of your middle finger with your thumb's tip
"""

# IMPORTANT
# Distance to determine perform click / drag or not
CLICK_MAX_DIST = 40
SCROLL_MAX_DIST = 30

# Order: Blue, Green, Red
DOT_COLOR = (0, 0, 255)
CONNECTION_COLOR = (0, 255, 0)

THUMB_TIP = 4
INEDX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12

EXIT_KEY = 'q'

def main():
    
    mpDrawing = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands

    res_1080 = (1920, 1080)
    res_720 = (1280, 720)

    cap = cv2.VideoCapture(0)
    # Scale it to 720p
    cap.set(3, res_720[0])
    cap.set(4, res_720[1])

    down = False

    with mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:

        while (cap.isOpened()):
            success, frame = cap.read()

            # Flip the frame horizontally
            image = cv2.flip(frame, 1)
            
            image, results = detectHands(hands, image)

            lm = []
            landmarks = results.multi_hand_landmarks
            if (landmarks):
                lm = formatLandmarks(landmarks, image, mpDrawing, mpHands, DOT_COLOR, CONNECTION_COLOR, True)

                # Index, FOR MOVING THE CURSOR
                x1, y1 = lm[INDEX_TIP]['x'], lm[INDEX_TIP]['y']
                x1, y1 = scaleCoords(x1, y1, res_1080)

                currY = y1

                # The coords has to be int
                try:
                    # This might cause error with no error message available = =
                    win32api.SetCursorPos((x1, y1))
                except:
                    print('ERROR when calling win32api.SetCursorPos()')

                # Thumb
                x0, y0 = lm[THUMB_TIP]['x'], lm[THUMB_TIP]['y']
                x0, y0 = scaleCoords(x0, y0, res_1080)

                # Another joint / node to perform click / hold / drag etc...
                x2, y2 = lm[INDEX_PIP]['x'], lm[INDEX_PIP]['y']
                x2, y2 = scaleCoords(x2, y2, res_1080)

                # Click
                dist = hypot(x2 - x0, y2 - y0)
                # print(dist)

                # Perform action with index finger's tip's coords
                if (dist <= CLICK_MAX_DIST):
                    # down = True
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x1, y1, 0, 0)
                else:
                    # down = False
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x1, y1, 0, 0)
                
                # print(dist)

            cv2.imshow('Monitor control via fingers', image)

            # Exit if user pressed 'q'
            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break
    
    print('--- End of Program ---')

def scaleCoords(x, y, resolution, returnInt=True):
    """
        Upscale the coordinate

        And fix it's boundaries (resolution ~ 0)
    """
    x, y = x * resolution[0], y * resolution[1]

    if (x > resolution[0]):
        x = resolution[0]
    elif (x < 0):
        x = 0
    
    if (y > resolution[1]):
        y = resolution[1]
    elif (y < 0):
        y = 0

    if not (returnInt):
        return x, y
    else:
        return int(x), int(y)

if (__name__ == '__main__'):
    main()