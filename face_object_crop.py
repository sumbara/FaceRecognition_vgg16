import cv2, os, pathlib

file_path = "F:\Data\PIRL_faceData"
cascade_file = "F:\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml"

# file_path 폴더내에 폴더들의 목록을 순차적으로 처리
for dirName in os.listdir(file_path) :
    name = dirName
    detectPath = 'F:\Data\PIRL_faceData_detect/' + name
    pathlib.Path(detectPath).mkdir(parents=True, exist_ok=True)

    # 폴더 목록 내 파일 목록을 순차적으로 처리
    for fileName in os.listdir(file_path + '/' + name) :
        image = cv2.imread(file_path + '/' + name + '/' + fileName)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cascade = cv2.CascadeClassifier(cascade_file)
        face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                                             ,flags=cv2.CASCADE_SCALE_IMAGE)

        if len(face_list) > 0 :
            for face in face_list :
                x, y, w, h = face
                #읽어낸 크기만큼 이미지를 잘라낸다
                cropped = image[y:y+h, x:x+w]
                # 파일 저장을 위해 경로 변경
                os.chdir(detectPath)
                # 얼굴만 잘라낸 파일 저장
                cv2.imwrite(fileName, cropped)
                # 다시 경로 변경
                os.chdir('../../')
        else :
            print ("no face")