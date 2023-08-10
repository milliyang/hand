import cv2
import mediapipe as mp

from hand_utils import detectHands

# Order: Blue, Green, Red
DOT_COLOR = (0, 0, 255)
CONNECTION_COLOR = (0, 255, 0)

EXIT_KEY = 'q'

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

            landmarks = results.multi_hand_landmarks
            if (landmarks):
                for hand in landmarks:
                    # Draws the detection on the image
                    mpDrawing.draw_landmarks(
                        image, hand, mpHands.HAND_CONNECTIONS,
                        mpDrawing.DrawingSpec(color=DOT_COLOR, thickness=2, circle_radius=4),
                        mpDrawing.DrawingSpec(color=CONNECTION_COLOR, thickness=2, circle_radius=2)
                    )
                    
                    # print(type(hand))

            cv2.imshow('Hand tracking', image)

            # print(image.shape)
            # cv2.imwrite('leo.png', image)
            # break

            # Exit if user pressed 'q'
            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break
    
    cap.release()
    cv2.destroyAllWindows()

    print("--- End of Program ---")

if (__name__ == '__main__'):
    main()