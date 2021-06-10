import os
import cv2
from glob import glob


def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")


def save_frame(video_path, save_dir, gap):
    name = video_path.split("/")[-1].split("\\")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name)
    create_dir(save_path)

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:  # meaning video end is reached and no ret val
            cap.release()
            break  # stop the loop bec no more frames
        if idx == 0:
            cv2.imwrite(f"{save_path}/{idx}.png", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{idx}.png", frame)

        idx += 1


if __name__ == "__main__":
    video_paths = glob("data/scene2/*")
    save_dir = "data/frames"

    for path in video_paths:
        save_frame(path, save_dir, gap=10)
