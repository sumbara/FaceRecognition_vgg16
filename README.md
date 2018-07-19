얼굴인식 시 vgg16 Fine Tuning을 통한 학습 내용입니다.

Image 스크레이핑 및 크롤링 과정부터 학습 방법까지 코드로 작성해 두었습니다.

스크레이핑 크롤링 : image_scrapper.py (_by_array는 배열로 받는것)
얼굴만 크롭핑 : face_object_crop
얼굴 데이터로 데이터셋 만들기 : face_to_dataSet
얼굴 데이터셋으로 학습하고 예측하기 : train_and_predict_model

ImageNet으로 pre-trained된 vgg16모델을 사용하기 위해 vgg16.npy 파일과
vgg16_trainable.py 와 utils.py 파일이 추가되어 있습니다.
