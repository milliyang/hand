import cv2

def blurFaces(img, newBox, sigmaX=10, sigmaY=10):

    topLeft = (newBox[0], newBox[1])
    rightBottom = (newBox[2], newBox[3])

    x, y = topLeft[0], topLeft[1]
    w, h = rightBottom[0], rightBottom[1]
    
    # Cut out the target part
    ROI = img[y:y + h, x:x + w]
    blur = cv2.GaussianBlur(ROI, (0, 0), sigmaX=sigmaX, sigmaY=sigmaY)
    
    # Put it back in
    img[y:y + h, x:x + w] = blur

    return img

def blurExceptFace(img, newBox, sigmaX=10, sigmaY=10):

    topLeft = (newBox[0], newBox[1])
    rightBottom = (newBox[2], newBox[3])

    x, y = topLeft[0], topLeft[1]
    w, h = rightBottom[0], rightBottom[1]

    # Save down the face area
    face = img[y : y + h, x : x + w]

    # Blur the whole image
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigmaX, sigmaY=sigmaY)

    # Replace the face part with un-blurred version
    img[y : y + h, x : x + w] = face

    return img

def pixelateFaces(img, newBox, pixelateSide=8):

    topLeft = (newBox[0], newBox[1])
    rightBottom = (newBox[2], newBox[3])

    x, y = topLeft[0], topLeft[1]
    w, h = rightBottom[0], rightBottom[1]
    
    # Cut out the target part
    ROI = img[y:y + h, x:x + w]
    roiH, roiW, _ = ROI.shape

    # Pixelate it by resizing it to NxN
    temp = cv2.resize(
        ROI,
        (pixelateSide, pixelateSide),
        interpolation=cv2.INTER_LINEAR
    )

    # Resize it back to the original size
    ROI = cv2.resize(temp, (roiH, roiW), interpolation=cv2.INTER_NEAREST)

    # Put it back into the original image
    try:
        img[y:y + h, x:x + w] = ROI
    
    except ValueError:
        pass

    # Why a try except & pass?
    # -
    # The following error will be thrown when the face is not INSIDE the image
    # E.g In the image's border
    """
    Traceback (most recent call last):
    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/pixelate_faces.py", line 84, in <module>
        main()
    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/pixelate_faces.py", line 43, in main
        image = pixelateFaces(image, newBox)
    File "c:/Users/User/Documents/GitHub Repositories/ManHinnn0509/mediapipe-scripts/face_detection/pixelate_faces.py", line 79, in pixelateFaces
        img[y:y + h, x:x + w] = ROI
    ValueError: could not broadcast input array from shape (183,182,3) into shape (182,183,3)
    [ WARN:0] global D:\a\opencv-python\opencv-python\opencv\modules\videoio\src\cap_msmf.cpp (438) `anonymous-namespace'::SourceReaderCB::~SourceReaderCB terminating async callback
    """

    return img
