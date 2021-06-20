import cv2
import mimetypes
import os
import shutil
import math
import random

# video = cv2.VideoCapture(path)

# print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
# print("Video length in frames: {0}".format(length))
# ms = length % fps
# secs = math.floor(length / fps)
# mins = math.floor(secs / 60)
# print("Video length: {0} mins {1} sec {2} ms".format(mins, secs % 60, ms))


# input dir path
# input_dataset_path = "./dataset/"
# # output dir path
# output_path = "./dataset_processed"
# # traverse the input dir
# cheater_clips = os.listdir(input_dataset_path + "cheating")
# not_cheater_clips = os.listdir(input_dataset_path + "not_cheating")
# # randomly select test data
# cheater_percentage = len(cheater_clips) * 20 / 100
# test_cheater_clips = random.sample(cheater_clips, math.floor(cheater_percentage))
# not_cheater_percentage = len(not_cheater_clips) * 20 / 100
# test_not_cheater_clips = random.sample(not_cheater_clips, math.floor(not_cheater_percentage))
# # create output dir or throw error if already exists
# if os.path.exists(output_path) is False:
#   os.mkdir(output_path)
# if len(os.listdir(output_path)) > 0:
#   raise FileExistsError("Provide a clean output directory")

# os.mkdir(output_path + "/test")
# for clip in test_cheater_clips:
#   shutil.copyfile("{0}cheating/{1}".format(input_dataset_path, clip), output_path + "/test/" + clip)

# for clip in test_not_cheater_clips:
#   shutil.copyfile("{0}not_cheating/very-good-players/{1}".format(input_dataset_path, clip), output_path + "/test/" + clip)

# print("Create test dataset with {0} clips with cheaters and {1} clips with players. Total: {2}"
#       .format(
#         len(test_cheater_clips),
#         len(test_not_cheater_clips),
#         len(test_cheater_clips) + len(test_not_cheater_clips)
#       ))

# remove frames to get an fps of 20
def downsample_frames(clip_name: str, target_fps = 20):
  clip = cv2.VideoCapture(clip_name)
  fps = clip.get(cv2.CAP_PROP_FPS)
  length = clip.get(cv2.CAP_PROP_FRAME_COUNT)

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # specify last param for greyscale
  out = cv2.VideoWriter('output.mp4', fourcc, target_fps, (640,  480), 0)

  frame_count = 0
  while clip.isOpened():
    ret, frame = clip.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break

    if length < fps:
      out.write(frame)
    else:
      if fps == 30 and frame_count % 3 == 2:
        out.write(frame)
      if fps == 60 and frame_count % 2 == 1:
        out.write(frame)
    frame_count += 1

  clip.release()
  out.release()

# convert frames to grayscale
# context stream: downsize frames to 89 x 89
# output file: name-context.mp4
# fovea stream: extract center of 89 x 89
# output file: name-fovea.mp4
# augument the dataset by flipping all data


# while video.isOpened():
#   ret, frame = video.read()
#   if not ret:
#     print("Can't receive frame (stream end?). Exiting ...")
#     break

#   # frame size needs to be equal with output video size
#   frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
#   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#   out.write(frame)
#   cv2.imshow('frame', frame)
#   if cv2.waitKey(1) == ord('q'):
#     break

# Release everything if job is finished
# video.release()
# out.release()
# cv2.destroyAllWindows()

# if os.path.isdir(path):  
#     print("\nIt is a directory")  
# elif os.path.isfile(path):
#   file_type, other = mimetypes.guess_type(path)
#   print("File type is {0}".format(file_type))