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
video_file='C:/Users/liangch/Desktop/MotionStyleHandData/runJogging_rgb.avi'
video_file='C:/Users/chliang/Desktop/realsense_python/kickSidekick_rgb.avi'
# video_file='C:/Users/chliang/Desktop/realsense_python/kickFrontkick_rgb.avi'
# video_file='C:/Users/chliang/Desktop/realsense_python/walkCrossover_rgb.avi'
# video_file='C:/Users/chliang/Desktop/realsense_python/walkInjured_rgb.avi'
# video_file='C:/Users/chliang/Desktop/realsense_python/runJogging_rgb.avi'
# video_file='C:/Users/chliang/Desktop/realsense_python/runSprint_rgb.avi'
# video_file='C:/Users/chliang/Desktop/realsense_python/runStride_rgb.avi'
# video_file = 'C:/Users/john8/Downloads/newRecord_2022_9_12/frontKickNew_rgb.avi'
# video_file = 'C:/Users/john8/Downloads/newRecord_2022_9_12/sideKickNew_rgb.avi'
# video_file = 'C:/Users/john8/Downloads/newRecord_2022_9_12/walkNormal_rgb.avi'
# video_file = 'C:/Users/john8/Downloads/newRecord_2022_9_12/walkIInjured_rgb.avi'
# video_file = 'C:/Users/john8/Downloads/newRecord_2022_9_14/jumpHurdle_rgb.avi'
video_file = 'C:/Users/john8/Downloads/newRecord_2022_9_14/jumpJoy_rgb.avi'


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

class DetectHandLM(): 
    def __init__(self) -> None:
        self.curDetectLM = None
        self.isCapturingLM = False
        self.videoFile = None

def captureByMediaPipe(videoFile, testingStageFunc, forOutputLM): 
    cap = cv2.VideoCapture(videoFile)
    tmpTime = time.time()
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5, 
        max_num_hands=1) as hands:

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
            # landmarks in a frame
            _landMarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                    if True: 
                        # _landMarks.extend(hand_landmarks.landmark)
                        # print(_landMarks[0])
                        handLMPred = [{'x': data_point.x, 'y': data_point.y, 'z': data_point.z} for data_point in hand_landmarks.landmark]
                        print(handLMPred)
                        result = testingStageFunc(handLMPred)
                        result = [{
                            "time": 0, 
                            "data": [{"x": dataArr[0, 0], "y": dataArr[0, 1], "z": dataArr[0, 2]} for dataArr in result]
                        }]
                        result = str(result)
                        result = result.replace('\'', '\"')
                        forOutputLM[0] = result
                        print(result)
                        # curTime = time.time()
                        # print('timeCost: ', curTime-tmpTime)
                        # tmpTime = curTime
                        print('-------')
                        # tmp_land_mark=hand_landmarks
                        # tmp_image=image
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            # print(len(_landMarks))
            # print('-------')
    cap.release()
    return 'EndCapture'

# Save to file, and serialize to a json file
if __name__ == '__main__': 
    cap = cv2.VideoCapture(video_file)
    # cap = cv2.VideoCapture(1)   # webcam
    detectLMs = []
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5, 
        max_num_hands=1) as hands:

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
            # landmarks in a frame
            _landMarks = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                    if True: 
                        detectLMs.append(
                            {
                                'time': time.time(), 
                                'data': hand_landmarks.landmark
                            }
                        )
                        # print([(data_point.x, data_point.y, data_point.z) for data_point in hand_landmarks.landmark])
                        # print('-------')
                        # tmp_land_mark=hand_landmarks
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
            print('-------')
            print(image.shape)
        # print(len(detectLMs))
        # print(len(detectLMs[0]['data']))

        # Serialize the hand landmarks in MediaPipe format. Serialize: [{'time': 0, 'data': [[1, 2, 3], ...]}, ...]
        for i in range(len(detectLMs)): 
            detectLMs[i]['data'] = [{'x': j.x, 'y': j.y, 'z': j.z} for j in detectLMs[i]['data']]
        import json
        # with open('./complexModel/walkInjured.json', 'w') as WFile: 
        with open('./complexModel/newRecord/jumpJoy_rgb.json', 'w') as WFile: 
            json.dump(detectLMs, WFile)
            
        # print(json.dumps(detectLMs))
    cap.release()

# Save image with hand landmarks
#cv2.imwrite('image_w_lm.jpg', tmp_image)

# Plot hand landmarks alone
# mp_drawing.plot_landmarks(
#         tmp_land_mark, mp_hands.HAND_CONNECTIONS, azimuth=5)