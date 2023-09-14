import cv2
import mediapipe as mp

from hand_utils import calcMidPoint, formatLandmarks, detectHands, drawLine, addText

"""
    Try to calculate the coords. of the palm and draw it out
"""

# Order: Blue, Green, Red
DOT_COLOR = (0, 0, 255)
CONNECTION_COLOR = (0, 255, 0)
LINE_COLOR = (255, 0, 255)
NODE_COLOR = (255, 255, 0)

EXIT_KEY = 'q'

WRIST = 0
MIDDLE_FINGER_MCP = 9

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
                drawLine(image, lm, WRIST, MIDDLE_FINGER_MCP, LINE_COLOR)

                """
                x1, y1 = lm[WRIST]['cx'], lm[WRIST]['cy']
                x2, y2 = lm[MIDDLE_FINGER_MCP]['cx'], lm[MIDDLE_FINGER_MCP]['cy']
                midPoint = calcMidPoint(x1, y1, x2, y2)

                # Pass thickness=-1 to fill the circle
                cv2.circle(image, midPoint, radius=5, color=NODE_COLOR, thickness=-1)
                """

            cv2.imshow('Hand tracking', image)

            # Exit if user pressed 'q'
            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break

    print("--- End of Program ---")


if (__name__ == '__main__'):
    main()