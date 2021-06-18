import cv2
import mimetypes

video = cv2.VideoCapture("./dataset_raw/10 Minutes of HACKER USING Aimbot & Wall Hack in Apex Legends Season 9!.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

file_type, other = mimetypes.guess_type("./dataset_raw/10 Minutes of HACKER USING Aimbot & Wall Hack in Apex Legends Season 9!.mp4")

print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
print("File type is {0}".format(file_type))

video.release()