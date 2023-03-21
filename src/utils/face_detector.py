"""
File : face_detector.py
Description : This file contains four main function:
              detect_face: Detects bounding boxes from the specified image.
              drawbox: Draw bounding boxes and frame number in the specified image.
              read_webcamFrame: Read frame from webcam, process frame and show frame.
              write_video:Create video writer object.
Created on : 3-Jan-2023
Author :Ankit Katiyar
E-mail :katiyar786ankit@gmail.com
"""
import cv2
import os
from src import constant

detector = cv2.CascadeClassifier(constant.FACE_DETECTOR_MODEL_PATH)


def detect_face(detector, img):
    """
        Detects bounding boxes from the specified image.
        param img: image to process
        return: list containing all the bounding boxes detected with their keypoints.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(
            gray,
            scaleFactor=constant.SCALE_FACOR,
            minNeighbors=constant.MIN_NEIGHBOUR,
            minSize=constant.MIN_SIZE,
        )
        return detections
    except Exception as e:
        print("Exception in detect_face: {}".format(e))


def drawbox(img, detections, num):
    """
        Draw bounding boxes and frame number in the specified image.
        param: img: image to process
               detections: List of bounding boxes
               num: frame number
        return: image after drawn bounding box and frame number.
    """
    try:
        for box in detections:
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x+w, y+h),
                          constant.COLOR, constant.SCALE)
            cv2.putText(img, 'Frame:'+str(num), constant.LOCATION,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, constant.COLOR, constant.THICKNESS, cv2.LINE_AA)
        return img
    except Exception as e:
        print("Exception in drawbox: {}".format(e))


def write_video():
    """
        Create video writer object.
        param: None
        return: object.
    """
    try:
        file_path = constant.ASSIGNMENT3_OUTPUT_PATH
        if os.path.exists(file_path):
            os.remove(file_path)
        vwriter = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(
            file_path, vwriter, constant.FPS, constant.RESOLUTION)
        return out
    except Exception as e:
        print("Exception in write_video: {}".format(e))


def read_webcamFrame():
    """
        Read frame from webcam, process frame and show frame.
        param: None
        return: None
    """
    try:
        video_writer = write_video()
        cap = cv2.VideoCapture(0)
        num = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            num += 1
            detections = detect_face(detector, frame)
            image = drawbox(frame, detections, num)
            video_writer.write(image)
            cv2.imshow('frame', image)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Exception in read_webcamFrame: {}".format(e))
