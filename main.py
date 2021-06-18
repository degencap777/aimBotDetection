import cv2
import mimetypes
import os
import math

path = "./dataset_raw/10 Minutes of HACKER USING Aimbot & Wall Hack in Apex Legends Season 9!.mp4"
video = cv2.VideoCapture(path)

fps = video.get(cv2.CAP_PROP_FPS)
length = video.get(cv2.CAP_PROP_FRAME_COUNT)

print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
print("Video length in frames: {0}".format(length))
ms = length % fps
secs = math.floor(length / fps)
mins = math.floor(secs / 60)
print("Video length: {0} mins {1} sec {2} ms".format(mins, secs % 60, ms))

video.release()

# if os.path.isdir(path):  
#     print("\nIt is a directory")  
# elif os.path.isfile(path):
#   file_type, other = mimetypes.guess_type(path)
#   print("File type is {0}".format(file_type))