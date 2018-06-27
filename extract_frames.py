import numpy
import cv2

path = 'harder_challenge_video.mp4'
vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()

count = 0
success = True

while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite('harder_challenge_video_frames/%d.jpg' % count, image)     # save frame as JPEG file
  count += 1
