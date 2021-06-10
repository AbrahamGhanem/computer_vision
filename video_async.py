import cv2
import asyncio
import numpy as np


class MultiCameraCapture:

    def __init__(self, sources: dict) -> None:
        assert sources
        print(sources)

        self.captures = {}
        for camera_name, link in sources.items():
            cap = cv2.VideoCapture(link)
            print(camera_name)
            assert cap.isOpened()  # if any of the cams can't be opened
            self.captures[camera_name] = cap  # store the camera name as an id and the opened capture obj inside of dict

    @staticmethod
    async def read_frame(capture):
        capture.grab()
        ret, frame = capture.retrieve()
        if not ret:
            print("empty frame")
            return
        return frame

    @staticmethod
    async def show_frame(window_name: str, frame: np.array):
        cv2.imshow(window_name, frame)

    # make an async generator to read the camera_name, the captures from the dict
    # it is needed for the usage of the async for loop
    async def async_camera_gen(self):
        for camera_name, capture in self.captures.items():
            yield camera_name, capture

