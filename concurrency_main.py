import cv2
import datetime
import asyncio
import signal
from glob import glob
from video_async import MultiCameraCapture
from utils.add_datetime import add_timestamp_to_frame
from utils.face_detection import run_face_detection
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial


async def run_fd_time(frame):
    task1 = asyncio.create_task(add_timestamp_to_frame(frame))
    await task1
    await asyncio.sleep(0.01)


async def run_blocking_func(loop_, frame):
    with ThreadPoolExecutor() as pool:
        blocking_func = partial(run_face_detection, frame)
        frame = await loop_.run_in_executor(pool, blocking_func)
        await asyncio.sleep(0.01)
    return frame


async def main(loop_, captured_obj):
    loop = asyncio.get_running_loop()
    while True:
        async for camera_name, cap in captured.async_camera_gen():
            frame = await asyncio.create_task(captured.read_frame(cap), name="frame_reader")

            await asyncio.create_task(run_fd_time(frame), name="add_timestamp")

            task1 = asyncio.create_task(captured_obj.show_frame(camera_name, frame), name="show_frame")
            task2 = asyncio.create_task(run_blocking_func(loop_, frame), name="face_detection")

            await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)

            if cv2.waitKey(1) == 27:
                break


async def shutdown_(signal_, loop_):
    """For a normal shutdown process."""

    print(f"Received signal -> {signal_ }")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    print(f"Cancelling {len(tasks)} tasks")
    await asyncio.gather(*tasks)

    loop_.stop()


if __name__ == "__main__":
    # if we had webcams we would have read the cameras from the JSON
    video_paths = glob("data/scene2/*")
    camera1 = video_paths[0]
    camera2 = video_paths[1]
    # cameras should be a dict with {"videoname": "path"}
    # {"video1_lap": camera1, "video1_mob": camera2} in JSON camera1,camera2 would be replaced with the cam OP num
    cameras = {"video1_lap": camera1, "video1_mob": camera2}
    captured = MultiCameraCapture(sources=cameras)
    loop = asyncio.get_event_loop()
    # Signal handler
    signals = {signal.SIGHUP, signal.SIGTERM, signal.SIGINT}
    for s in signals:
        loop.add_signal_handler(
            s, lambda f=s: asyncio.create_task(shutdown_(s, loop)))
    try:
        loop.run_until_complete(main(loop_=loop, captured_obj=captured))
    finally:
        print("Successfully shutdown service")
        loop.close()
