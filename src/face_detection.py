import cv2
import os


def catch_video(window_name):
    cap = cv2.VideoCapture(0)  # 指明使用的摄像头
    classifier = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
    rectangle_color = (0, 255, 0)  # bgr

    while cap.isOpened():
        ret, frame = cap.read()  # 分别返回读取是否成功，读取返回的视频帧
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将帧转化为灰度图

        # 使用OpenCV的算法检测所有面部
        face_rects = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))

        if len(face_rects) > 0:
            for face_rect in face_rects:
                x, y, w, h = face_rect  # 返回框选面部的方框的x，y轴坐标，x轴方向长度和y轴方向高度
                # 在每一帧画出方框，下面方法参数分别是帧，坐标，边长，颜色，线条粗细
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), rectangle_color, 2)

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c is ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_train_data(window_name, data_size, data_save_path):
    cap = cv2.VideoCapture(0)
    classfier = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
    rectangle_color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = classfier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(32, 32))
        if len(face_rects) > 0:
            for face_rects in face_rects:
                x, y, w, h = face_rects

                img_name = "%s%d.jpg" % (data_save_path, num)
                image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
                cv2.imwrite(img_name, image)
                num += 1
                font = cv2.FONT_HERSHEY_TRIPLEX
                cv2.putText(frame, "data_saved: %d/%d" % (num, data_size), (x, y - 30), font, 0.7, rectangle_color, 1)

                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), rectangle_color, 1)

        if num > data_size:
            break

        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # catch_video("Face Recognization v1.0")
    get_train_data("Get Face data", 100, os.getcwd() + '/myData/')
