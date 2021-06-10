# pip install pytube
from pytube import YouTube
import os
# where to save
SAVE_PATH = os.getcwd()

# link of the video to be downloaded
link = "https://www.youtube.com/watch?v=PJ5xXXcfuTc"
yt = YouTube(link)
stream = yt.streams.first()
stream.download(SAVE_PATH)