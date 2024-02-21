import cv2
import mediapipe as mp
import math
import socket

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 50002)

def check_pose(right_shoulder, right_elbow, right_wrist):
    elbow_x_similarity = abs(right_elbow.x - right_wrist.x) < 0.05
    elbow_angle = math.degrees(math.acos((right_shoulder.x - right_elbow.x) /
                                         math.sqrt((right_shoulder.x - right_elbow.x) ** 2 +
                                                   (right_shoulder.y - right_elbow.y) ** 2)))
    wrist_y_similarity = abs(right_elbow.y - right_wrist.y) < 0.05

    if elbow_x_similarity and 80 <= elbow_angle <= 100 and wrist_y_similarity:
        return 'Correct Pose'
    else:
        return 'Incorrect Pose'

def main():
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture("http://192.168.50.27:8080/?action=stream")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            pose_text = check_pose(right_shoulder, right_elbow, right_wrist)
            cv2.putText(frame, pose_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            plist = []
            plist.append(pose_text)
            if(pose_text == "Correct Pose"):
                count += 1
                if(count ==50):
                    break
            else:
                count = 0
            plist.append(count)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                ih, iw, _ = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                plist.append(x)
                plist.append(y)


        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        print(plist)
        sock.sendto(str.encode(str(plist)), serverAddressPort)


        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

