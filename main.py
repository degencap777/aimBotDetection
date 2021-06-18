import cv2

video = cv2.VideoCapture("./dataset_raw/10 Minutes of HACKER USING Aimbot & Wall Hack in Apex Legends Season 9!.mp4")

fps = video.get(cv2.CAP_PROP_FPS)

print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

video.release()