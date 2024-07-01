# NOT WHAT WE WANT
import cv2 
import mediapipe as mp 

mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (212, 130, 173)
def coordinate(id, h, w, img):
    # calculates abs value of the landmark points
    cx, cy = int(id.x*w), int(id.y*h)
    cv2.circle(img, (int(cx), int(cy)), 5, (255,255,255), cv2.FILLED)
    return cx, cy

hasprint = True
while cam.isOpened():
    ret, frame = cam.read()
    
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    h, w, c, = frame.shape
    handsup = 0 # hands not up
    thumbs_correct = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            if(hasprint):
                print(hand_landmarks)
                hasprint = False
            for id, landmark in enumerate(hand_landmarks.landmark):
                if id == 0:
                    cx_0, cy_0 = coordinate(landmark, h, w, frame)

                if id == 10:
                    _, cy_10 = coordinate(landmark, h, w, frame)
                
                if id == 2:
                    _, cy_2 = coordinate(landmark, h, w, frame)

                if id == 3:
                    _, cy_3 = coordinate(landmark, h, w, frame)
                

                if id == 5:
                    _, cy_5 = coordinate(landmark, h, w, frame)

                if id == 9:
                    _, cy_9 = coordinate(landmark, h, w, frame)
                
                if id == 13:
                    _, cy_13 = coordinate(landmark, h, w, frame)

                if id == 17:
                    _, cy_17 = coordinate(landmark, h, w, frame)
                
                if id == 8:
                    _, cy_8 = coordinate(landmark, h, w, frame)

                if id == 12:
                    _, cy_12 = coordinate(landmark, h, w, frame)
                
                if id == 16:
                    _, cy_16 = coordinate(landmark, h, w, frame)

                if id == 20:
                    _, cy_20 = coordinate(landmark, h, w, frame)



            # check if hands is up, is the y10 greater or lower
            if cy_10 < cy_0:
                handsup = 1
                
                # cv2.putText(frame, "Hand Up", (cx_0, cy_0), cv2.FONT_HERSHEY_DUPLEX,
                #             1, (212,130,173), 1, cv2.LINE_AA)
            else:
                handsup = 0
                # cv2.putText(frame, "Hand Down", (cx_0, cy_0), cv2.FONT_HERSHEY_DUPLEX,
                #             1, (212,130,173), 1, cv2.LINE_AA)


            # forces hands to be straight up
            if (cy_2 > cy_10 and cy_2 < cy_0) and (cy_3 > cy_10 and cy_3 < cy_0):
                thumbs_correct = 1

                # cv2.putText(frame, "Thumbs Correct", (cx_0, cy_0), cv2.FONT_HERSHEY_DUPLEX,
                #             1, (212,130,173), 1, cv2.LINE_AA)
            else:
                thumbs_correct = 0
                # cv2.putText(frame, "Thumbs InCorrect", (cx_0, cy_0), cv2.FONT_HERSHEY_DUPLEX,
                #             1, (212,130,173), 1, cv2.LINE_AA)


            if (cy_5 < cy_8) and (cy_9 < cy_12) and (cy_13 < cy_16) and (cy_17 < cy_20):
                fingers_correct = 1
            else:
                fingers_correct = 0

            if handsup == 1 and thumbs_correct == 1 and fingers_correct == 1:
                cv2.putText(frame, "Action Performed", (cx_0, cy_0), cv2.FONT_HERSHEY_DUPLEX,            
                            1, (212,130,173), 1, cv2.LINE_AA)
                

            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # # Extract landmarks for both hands
            # landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            # # Here, you would add the logic to process these landmarks
            # # and extract the necessary features for classification

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()