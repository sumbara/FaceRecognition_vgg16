얼굴인식 시 vgg16 Fine Tuning을 통한 학습 내용입니다.

Image 스크레이핑 및 크롤링 과정부터 학습 방법까지 코드로 작성해 두었습니다.

스크레이핑 크롤링 : image_scrapper.py (_by_array는 배열로 받는것)
얼굴만 크롭핑 : face_object_crop
얼굴 데이터로 데이터셋 만들기 : face_to_dataSet
얼굴 데이터셋으로 학습하고 예측하기 : train_and_predict_model

ImageNet으로 pre-trained된 vgg16모델을 사용하기 위해 vgg16.npy 파일과
vgg16_trainable.py 와 utils.py 파일이 추가되어 있습니다.

web_cam 폴더는 캠을 설치하여 사용할 경우 사용할 수 있는 코드들이 있습니다.

face_tracking_saver.py : 얼굴을 추적해서 사용자의 로컬 폴더에 얼굴 사진을 저장
predict.py : train_and_predict_model.py를 통해서 학습한 카테고리들을 토대로 predict_face.py가 넘겨준 얼굴이 누구의 얼구인지 예측합니다. 예측 결과는 string 으로 return 합니다.
predict_face.py : 얼굴을 추적하면서 실시간으로 얼굴이 학습한 모델을 토대로 누군지 알려준다.
