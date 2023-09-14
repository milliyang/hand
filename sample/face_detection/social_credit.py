import cv2
import mediapipe as mp

"""
    JUST FOR FUN

    Displays your social credit scores 🥶🥵🥶🥵🥶🥵
"""

EXIT_KEY = 'q'

def main():
    res_720 = (1280, 720)

    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    # Scale it to 720p
    cap.set(3, res_720[0])
    cap.set(4, res_720[1])
    
    with mpFaceDetector.FaceDetection(min_detection_confidence=0.7) as faceDetection:
        
        while (cap.isOpened()):
            success, frame = cap.read()

            # Camera's width & height
            width  = int(cap.get(3))
            height = int(cap.get(4))

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)

            # Convert the color for the program to process
            # Since cv2 uses BGR and mediapipe uses RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(image)

            # Convert it back for displaying after processing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if (results.detections):
                for id, detection in enumerate(results.detections):
                    # Draw the box around the face with built-in function
                    # mpDraw.draw_detection(image, detection)

                    # The box around the face
                    box = detection.location_data.relative_bounding_box
                    
                    # Scale the box with current cap's width and height
                    newBox = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)

                    # Draw the scaled box with cv2.rectangle
                    cv2.rectangle(image, newBox, (0, 0, 255), 3)

                    cv2.putText(
                        # image, f'Score: {detection.score[0]:.2f}',
                        image, f'Social credit: {-99999999:,}',
                        (newBox[0], newBox[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )

            cv2.imshow('Face detection', image)

            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break

if (__name__ == '__main__'):
    main()