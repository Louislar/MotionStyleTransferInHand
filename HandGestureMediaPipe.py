import cv2
import mediapipe as mp
import asyncio
import time

now = lambda: time.time()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands   

# Read video

video_file='C:/Users/liangch/Desktop/V_20211210_113019_OC0.mp4'
video_file='C:/Users/liangch/Desktop/V_20211224_233134_OC0.mp4'
video_file='C:/Users/liangch/Desktop/V_20211225_094631_OC0.mp4'
video_file='C:/Users/liangch/Desktop/hand_gesture/1far_near_rgb.avi'
video_file='C:/Users/liangch/Desktop/hand_gesture/3left_right_rgb.avi'
# video_file=1
tmp_counter = 0
tmp_land_mark = None
tmp_image = None

# async def print123func():
#   tmp_num=0
#   while tmp_num<=5: 
#     print(tmp_num)
#     tmp_num+=1
#     await asyncio.sleep(1)

# if __name__ == "__main__":
#     start = now()
#     tasks = [print123func() for i in range(5)]
#     asyncio.run(asyncio.wait(tasks))
#     print('TIME: ', now() - start)


cap = cv2.VideoCapture(video_file)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

            if True: 
                print([(data_point.x, data_point.y, data_point.z) for data_point in hand_landmarks.landmark])
                print('-------')
                tmp_land_mark=hand_landmarks
                tmp_image=image
            tmp_counter+=1
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

# Save image with hand landmarks
#cv2.imwrite('image_w_lm.jpg', tmp_image)

# Plot hand landmarks alone
# mp_drawing.plot_landmarks(
#         tmp_land_mark, mp_hands.HAND_CONNECTIONS, azimuth=5)