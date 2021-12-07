import cv2
import glob
from person_tracking import person_tracker


def frames_to_video(frames_path, video_save_path):
#This function creates a video from the frames saved in the path given as argument and
#save the created video in video_save_path
    img_array = []
    for image_file in glob.glob(frames_path):
        img = cv2.imread(image_file)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':

    input_video_path = 'basket_test.mp4'
    output_video_path = 'basket_result.avi'
    output_frames_path = 'output/*.jpg'

    person_tracker(input_video_path)
    frames_to_video(output_frames_path, output_video_path)








