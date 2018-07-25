import cv2
import os
import sys
import tensorflow as tf
import numpy as np

import vgg16_trainable as vgg16
import predict

import multiprocessing as mp
from multiprocessing.pool import  ThreadPool

CAM_ID = 0

# 폰트 등록
font = cv2.FONT_HERSHEY_SIMPLEX

# 추적기능 상태
# 얼굴 인식
TRACKING_STATE_CHECK = 0
# 얼굴인식 위치를 기반으로 추적 기능 초기화
TRACKING_STATE_INIT = 1
# 추적 동작
TRACKING_STATE_ON = 2

name = 'predicting..'
img_size = 224
root_dir = "F:\Data\FaceTest_manyFile"
category = os.listdir(root_dir)

def predict_from_pretrained(image):
    # n_class = len(os.listdir(root_dir))
    test_batch = []
    img = resize_with_pad(image)
    img = img.reshape((1, 224, 224, 3))

    data = np.asarray(img)

    prob = sess.run(vgg.prob, feed_dict = {train_mode: False, images: data})
    #print(prob)
    prob_argmax = np.argmax(prob, axis=1)
    print(prob_argmax)
    print("predict: ", category[prob_argmax[0]])

    return category[prob_argmax[0]]

def resize_with_pad(image, height=224, width=224):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


if __name__ == '__main__':

    n_class = 997

    # with tf.Session() as sess:
    #     images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    #     train_mode = tf.placeholder(tf.bool)
    #     vgg = vgg16.Vgg16('./test-save_pirl.npy', trainable = False, n_class = n_class)
    #
    #     with tf.name_scope("content_vgg"):
    #     vgg.build(images, train_mode, int_image=True)

    sess = tf.Session()
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    train_mode = tf.placeholder(tf.bool)
    vgg = vgg16.Vgg16('./test-save_pirl_2.npy', trainable=False, n_class=n_class)

    vgg.build(images, train_mode, int_image=True)


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    if not cap.isOpened():
        print("Could not open video")
        sys.exit()

    cascade_path = 'F:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml'
    cascade = cv2.CascadeClassifier(cascade_path)

    # 추적 상태 저장용 변수
    TrackingState = 0
    # 추적 영역 저장용 변수
    TrackingROI = (0, 0, 0, 0)

    frame_count = 0
    p1_temp = [0, 0]

    while True:
        _, frame = cap.read()

        if not _:
            break

        # 추적 상태가 얼굴 인식이면 얼굴 인식 기능 동작
        # 처음에 무조건 여기부터 들어옴
        if TrackingState is TRACKING_STATE_CHECK:
            # 흑백 변경
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 히스토그램 평활화(재분할)
            grayframe = cv2.equalizeHist(grayframe)
            # 얼굴 인식
            faces = cascade.detectMultiScale(grayframe, 1.1, 5, 0, (30, 30))

            # 얼굴이 1개라도 잡혔다면
            if len(faces) > 0:
                # 얼굴 인식된 위치 및 크기 얻기
                x, y, w, h = faces[0]
                # 인식된 위치및 크기를 TrackingROI에 저장
                TrackingROI = (x, y, w, h)
                # 인식된 얼굴 표시 순식간에 지나가서 거의 볼수 없음(녹색)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3, 4, 0)
                # 추적 상태를 추적 초기화로 변경
                TrackingState = TRACKING_STATE_INIT

        # 추적 초기화
        elif TrackingState is TRACKING_STATE_INIT:
            tracker = cv2.TrackerKCF_create()
            # 추적 함수 초기화
            # 얼굴인식으로 가져온 위치와 크기를 함께 넣어준다.
            ok = tracker.init(frame, TrackingROI)
            if ok:
                # 성공하였다면 추적 동작상태로 변경
                TrackingState = TRACKING_STATE_ON
                # print('tracking init succeeded')
            else:
                # 실패하였다면 얼굴 인식상태로 다시 돌아감
                TrackingState = TRACKING_STATE_CHECK
                print('tracking init failed')

        # 추적 동작
        elif TrackingState is TRACKING_STATE_ON:
            # 추적
            ok, TrackingROI = tracker.update(frame)
            if ok:
                # 추적 성공했다면
                p1 = (int(TrackingROI[0]), int(TrackingROI[1]))
                p2 = (int(TrackingROI[0] + TrackingROI[2]), int(TrackingROI[1] + TrackingROI[3]))

                cv2.rectangle(frame, p1, p2, (255, 255, 255), 2, 1)
                cv2.putText(frame, name, (p1[0] - 5, p1[1] - 5), font, 0.9, (255, 255, 0), 2)

                frame_count += 1
                if frame_count is 40:
                    if p1_temp[0] is 0:     # 맨 처음
                        p1_temp[0] = p1[0]
                        p1_temp[1] = p1[1]
                        frame_count = 0
                        name = 'predicting..'
                    elif p1[0] is p1_temp[0]:
                        print('Tracking failed')
                        p1_temp = [0, 0]
                        TrackingROI = (0, 0, 0, 0)
                        tracker = None
                        frame_count = 0
                        name = 'tracking..'
                        TrackingState = TRACKING_STATE_CHECK
                    else:
                        p1_temp[0] = p1[0]
                        p1_temp[1] = p1[1]

                        image = frame[p1[1]: p2[1], p1[0]: p2[0]]
                        if name is 'predicting..':
                            result = predict_from_pretrained(image)
                            name = result

                        frame_count = 0

            else:
                print('Tracking failed')
                name = 'tracking..'
                p1_temp = [0, 0]
                TrackingROI = (0, 0, 0, 0)
                tracker = None
                frame_count = 0
                TrackingState = TRACKING_STATE_CHECK

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(100)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

