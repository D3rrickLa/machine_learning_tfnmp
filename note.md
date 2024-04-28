## problem
is this a video or image dataset?



### how it could work - gesture
is this gesture recgonition or video recgonition?

with a gesture style dataset we can record the hand movement, basing the position on a known frame (640 x 470 for example) and that captured data we can label it as X (X being what that gesture is).


1. use mediapipe to capture the hand gesture position
2. translate that information into something useful
3. take that info and put it into a dataset folder - thinking EXCEL file or something
4. use tensorflow to build the model

### how it could work - video
1. use opencv to capture webcam data
2. somehow extract the frames - frames should be in bunches, like every n seconds
3. build a model from that bunched frame - can't use MediaPipe
4. hope it works

https://github.com/GibranBenitez/IPN-hand