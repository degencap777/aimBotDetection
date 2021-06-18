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

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# specify last param for greyscale
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,  480), 0)

while video.isOpened():
  ret, frame = video.read()
  if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    break

  # frame size needs to be equal with output video size
  frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  out.write(frame)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) == ord('q'):
    break

# Release everything if job is finished
video.release()
out.release()
cv2.destroyAllWindows()

# if os.path.isdir(path):  
#     print("\nIt is a directory")  
# elif os.path.isfile(path):
#   file_type, other = mimetypes.guess_type(path)
#   print("File type is {0}".format(file_type))