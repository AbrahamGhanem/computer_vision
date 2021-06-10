from moviepy.editor import *

myvideo1 = VideoFileClip(r'C:\Users\GHANEM\Desktop\OpenCV\data\highway.MP4')
myvideo1edited = myvideo1.subclip(53,57)  # from second 6 to 11
myvideo1edited.write_videofile(('highway_short.MP4'), codec='libx264')