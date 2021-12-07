import cv2
import glob
from person_tracking import person_tracker


def frames_to_video(frames_path):
    img_array = []
    for image_file in glob.glob(frames_path):
        img = cv2.imread(image_file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('basket_result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':

    person_tracker('basket_test.mp4')






