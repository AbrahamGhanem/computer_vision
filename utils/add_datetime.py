import cv2
import datetime
import asyncio


async def add_timestamp_to_frame(frame):
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    dt = str(datetime.datetime.now())
    cv2.putText(frame, dt,
                (10, 100),
                font, 1,
                (210, 155, 155),
                4, cv2.LINE_8)
    await asyncio.sleep(0.1)