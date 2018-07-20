import cv2
import sys
import os

# 카메라 번호
# /dev/video0 = 0
# 이곳에 동영상 파일명의 위치를 넣어주면 동영상 재생으로 동작함
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


def crop_image(p1, p2, frame):
    name_count = '1'.rjust(4, '0')

    if len(os.listdir('F:\Data\PIRL_faceData/180720/')) is not 0:
        last_file = os.listdir('F:\Data\PIRL_faceData/180720/')[-1]
        last_num = last_file.split('.')[0]
        name_count = str(int(last_num) + 1).rjust(4, '0')

    cv2.imwrite('F:\Data\PIRL_faceData/180720/' + name_count + '.jpg',
                frame[p1[1]:p2[1], p1[0]:p2[0]], params=[cv2.IMWRITE_JPEG_QUALITY, 100])


if __name__ == '__main__':

    # 카메라 열기
    video = cv2.VideoCapture(CAM_ID)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    # 카메라가 정상적으로 열리지 않았다면 프로그램 종료
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # 얼굴인식 함수 생성
    face_cascade = cv2.CascadeClassifier()
    # 얼굴인식용 haar 불러오기
    face_cascade.load('F:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml')

    # 추적 상태 저장용 변수
    TrackingState = 0
    # 추적 영역 저장용 변수
    TrackingROI = (0, 0, 0, 0)

    frame_count = 0
    p1_temp = [0, 0]

    # 프로그램 시작
    while True:
        # 카메라에서 1 frame 읽어오기
        ok, frame = video.read()
        # 영상이 없다면 프로그램 종료를 위해 while 문 빠져나옴
        if not ok:
            break

        # 추적 상태가 얼굴 인식이면 얼굴 인식 기능 동작
        # 처음에 무조건 여기부터 들어옴
        if TrackingState is TRACKING_STATE_CHECK:
            # 흑백 변경
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 히스토그램 평활화(재분할)
            grayframe = cv2.equalizeHist(grayframe)
            # 얼굴 인식
            faces = face_cascade.detectMultiScale(grayframe, 1.1, 5, 0, (30, 30))

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
                # 화면에 박스로 표시 (파랑)
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv2.putText(frame, 'Detected Face', (p1[0] - 5, p1[1] - 5), font, 0.9, (255, 255, 0), 2)

                frame_count += 1

                if frame_count is 15:
                    if p1_temp[0] is 0:
                        p1_temp[0] = p1[0]
                        p1_temp[1] = p1[1]
                        frame_count = 0
                    elif abs(p1[0] - p1_temp[0]) == 0:
                        print('Tracking failed')
                        p1_temp = [0, 0]
                        TrackingROI = (0, 0, 0, 0)
                        tracker = None
                        frame_count = 0
                        TrackingState = TRACKING_STATE_CHECK
                    else:
                        p1_temp[0] = p1[0]
                        p1_temp[1] = p1[1]
                        crop_image(p1, p2, frame)
                        frame_count = 0
            else:
                print('Tracking failed')
                p1_temp = [0, 0]
                TrackingROI = (0, 0, 0, 0)
                tracker = None
                frame_count = 0
                TrackingState = TRACKING_STATE_CHECK

        # 추적된 박스가 있으면 같이 표시됨
        cv2.imshow("Tracking", frame)

        # ESC키를 누르면 break로 종료
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
