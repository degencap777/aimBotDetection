import cv2
import mimetypes
import os

path = "./dataset_raw/10 Minutes of HACKER USING Aimbot & Wall Hack in Apex Legends Season 9!.mp4"
video = cv2.VideoCapture(path)

fps = video.get(cv2.CAP_PROP_FPS)

print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

if os.path.isdir(path):  
    print("\nIt is a directory")  
elif os.path.isfile(path):
  file_type, other = mimetypes.guess_type(path)
  print("File type is {0}".format(file_type))

video.release()