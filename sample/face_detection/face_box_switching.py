import cv2
import mediapipe as mp

"""
    Swapping 2 faces' box area

    Just for fun but This is horrible D:
"""

EXIT_KEY = 'q'

def main():
    mpFaceDetector = mp.solutions.face_detection
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
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

                if (len(results.detections) == 2):
                    # Select out the information of the 2 faces that were found
                    first = results.detections[0]
                    second = results.detections[1]

                    box1 = first.location_data.relative_bounding_box
                    box2 = second.location_data.relative_bounding_box

                    box1 = int(box1.xmin * width), int(box1.ymin * height), int(box1.width * width), int(box1.height * height)
                    box2 = int(box2.xmin * width), int(box2.ymin * height), int(box2.width * width), int(box2.height * height)

                    topLeft1 = (box1[0], box1[1])
                    rightBottom1 = (box1[3], box1[3])

                    x1, y1 = topLeft1[0], topLeft1[1]
                    w1, h1 = rightBottom1[0], rightBottom1[1]

                    topLeft2 = (box2[0], box2[1])
                    rightBottom2 = (box2[3], box2[3])

                    x2, y2 = topLeft2[0], topLeft2[1]
                    w2, h2 = rightBottom2[0], rightBottom2[1]

                    # Extrace the face box area
                    face1 = image[y1 : y1 + h1, x1 : x1 + w1]
                    face2 = image[y2 : y2 + h2, x2 : x2 + w2]

                    # Save the size of them
                    f1H, f1W, _ = face1.shape
                    f2H, f2W, _ = face2.shape

                    # Resize them into each other's size
                    face1 = cv2.resize(face1, (f2H, f2W), interpolation=cv2.INTER_LINEAR)
                    face2 = cv2.resize(face2, (f1H, f1W), interpolation=cv2.INTER_LINEAR)

                    # Exception might be thrown from the 2 above lines (cv2.resize)
                    """
                    Traceback (most recent call last):
                    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/face_box_switching.py", line 98, in <module>
                        main()
                    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/face_box_switching.py", line 64, in main
                        face1 = cv2.resize(face1, (f2H, f2W), interpolation=cv2.INTER_LINEAR)
                    cv2.error: OpenCV(4.5.4-dev) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4054: error: (-215:Assertion failed) inv_scale_x > 0 in function 'cv::resize'
                    """

                    # Switch them
                    image[y1 : y1 + h1, x1 : x1 + w1] = face2
                    image[y2 : y2 + h2, x2 : x2 + w2] = face1

                else:
                    # Default
                    for id, detection in enumerate(results.detections):
                        # Draw the box around the face with built-in function
                        # mpDraw.draw_detection(image, detection)

                        # The box around the face
                        box = detection.location_data.relative_bounding_box
                        
                        # Scale the box with current cap's width and height
                        newBox = int(box.xmin * width), int(box.ymin * height), int(box.width * width), int(box.height * height)

                        # Draw the scaled box with cv2.rectangle
                        cv2.rectangle(image, newBox, (0, 255, 0), 1)

                        cv2.putText(
                            image, f'Score: {detection.score[0]:.2f}',
                            (newBox[0], newBox[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                        )

            cv2.imshow('Face box swaping', image)

            if (cv2.waitKey(10) & 0xFF == ord(EXIT_KEY)):
                break

if (__name__ == '__main__'):
    main()