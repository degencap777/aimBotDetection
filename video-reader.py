import sys
import cv2

if __name__ == "__main__":
  video_name = sys.argv[1]
  delay = sys.argv[2] if sys.argv[2] else 500
  if not video_name:
    raise ValueError("Please provide a video path as first arg!")

  clip = cv2.VideoCapture(video_name)
  fps = clip.get(cv2.CAP_PROP_FPS)
  length = clip.get(cv2.CAP_PROP_FRAME_COUNT)

  print("Total frames: {0}, fps: {1}".format(length, fps))

  while clip.isOpened():
    ret, frame = clip.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
    
    cv2.imshow("clip_name", frame)
    if cv2.waitKey(delay) == ord('q'):
      break

  clip.release()
  cv2.destroyAllWindows()