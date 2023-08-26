import cv2
from PIL import Image
from fastai.vision import *
import warnings

warnings.simplefilter('ignore')


def live_video(camera_port=0):
    """
    Opens a window with live video.
    :param camera:
    :return:
    """

    # Open video on a give camera port
    # Export your model
    export = "C:/Users/Esada Licina/UNI.lu BICS/BICS sem2/Technic Part BSP2/BSP2-Technical_part_Python/Classification/export/"
    learn = load_learner(export)
    video_capture = cv2.VideoCapture(camera_port)
    # Set video frame properties
    # video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
    # video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cv2.namedWindow("Flower detection with fastai", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Flower detection with fastai", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    detect = False
    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        height, width, channels = frame.shape
        upper_left = (int(width / 4), int(height / 4))
        bottom_right = (int(width * 3 / 4), int(height * 3 / 4))

        # draw in the image
        frame_img = cv2.rectangle(frame, upper_left, bottom_right, (100, 100, 255), 2)

        # Display the resulting frame
        rect_img = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]
        cv2.imshow('Selection', rect_img)

        # start the classification
        if cv2.waitKey(1) == ord('s'):
            detect = not detect

        pred = ''
        if detect:
            # transform the frame into a image
            img = Image(pil2tensor(rect_img, dtype=np.float32).div_(255))
            pred = learn.predict(img)[0]

        # stop the classification
        if cv2.waitKey(1) == ord('f'):
            detect = False

        # Write description above pink rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)
        stroke = 2
        x = 200
        y = 100
        cv2.putText(frame_img, str(pred), (x, y), font, 1, color, stroke, cv2.LINE_AA)
        cv2.imshow("Flower detection with fastai", frame_img)

        if cv2.waitKey(1) == ord('q'):
            break

    # When everything is done, release the capture

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_video()
