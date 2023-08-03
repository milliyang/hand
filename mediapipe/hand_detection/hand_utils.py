import cv2

from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def getAudioDevice():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    return volume

def detectHands(hands, frame):
    # Recoloring the frame read from webcam
    # Default 'image' get from cv2 is in BGR
    # Convert it to RGB is for mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # Change it back to BGR after the process
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results

def calcMidPoint(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def addText(image, landmarks, text, p1Index, p2Index, textColor):
    """
        Add text that shows the volume near the middle of the thumb's tip and index finger's tip
    """
    x1, y1 = landmarks[p1Index]['cx'], landmarks[p1Index]['cy']
    x2, y2 = landmarks[p2Index]['cx'], landmarks[p2Index]['cy']

    midPoint = calcMidPoint(x1, y1, x2, y2)
    cv2.putText(image, text, midPoint, cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 1, cv2.LINE_AA)

def drawLine(image, landmarks, p1Index, p2Index, lineColor):
    """
        Draw line between thumb's tip and index finger's tip
    """
    x1, y1 = landmarks[p1Index]['cx'], landmarks[p1Index]['cy']
    x2, y2 = landmarks[p2Index]['cx'], landmarks[p2Index]['cy']

    # cv2.line(image, start_point, end_point, color, thickness)
    cv2.line(image, (x1, y1), (x2, y2), lineColor, 3)

def formatLandmarks(landmarks, image, mpDrawing, mpHands, dotColor, connectionColor, drawOnImage=True):
    """
        Draws hand on image,
        Returns a list of formatted landmarks with following format:

        [
            {
                'id': NODE_ID,
                'cx': int(mark.x * image.shape.width),
                'cy': int(mark.y * image.shape.height)
            },
            ...
        ]
    """
    l = []

    for handLandmark in landmarks:

        for id, mark in enumerate(handLandmark.landmark):
            # Draws the detection on the image
            if (drawOnImage):
                mpDrawing.draw_landmarks(
                    image, handLandmark, mpHands.HAND_CONNECTIONS,
                    mpDrawing.DrawingSpec(color=dotColor, thickness=2, circle_radius=4),
                    mpDrawing.DrawingSpec(color=connectionColor, thickness=2, circle_radius=2)
                )

            height, width, ignored = image.shape
            cx, cy = int(mark.x * width), int(mark.y * height)

            l.append(
                {
                    'id': id,
                    'cx': cx,
                    'cy': cy,
                    'x': mark.x,
                    'y': mark.y
                }
            )
    
    return l