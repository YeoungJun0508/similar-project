#닮은 연예인 찾기.

### 1.데이터베이스에 이미지 넣기.

Image Crawling.

출처:https://github.com/YoongiKim/AutoCrawler

네이버와 구글에서 해당 키워드를 검색했을 때 
노출되는 사진을 모두 긁어온다.

문제점:다른 연예인과 함께 가져오는 사진을 분리해야함.

>ex) 김종국 키워드로 검색했지만 다른 연예인 사진이 같이 오는 경우.
![image](https://github.com/YeoungJun0508/similar-project/assets/145903037/41a3bb30-3d64-4b75-9838-8a5632247b37)

구글 api 사용법: https://twinw.tistory.com/199

구글 api를 사용하여 얼굴만 잘라오는 코드: https://github.com/bwcho75/facerecognition/blob/master/com/terry/face/extract/crop_face.py

### 2.이미지 분류

2019년에 나온 이미지 분류 모델인 EfficientNet를 사전 훈련 모델로 사용.



