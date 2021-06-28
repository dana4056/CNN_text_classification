### CNN_text_classification
A program that uses CNN to detect alphabet cards and recognize letters in videos. 
<br>CNN 모델을 사용하여 영상에서 알파벳 카드를 감지하고 글자를 인식하는 프로그램입니다. 
<br><br><br>

### 파일구조
```bash
├── Data                        
│   ├── test_dataset.zip
│   └── train_dataset.zip
├── cnn.py                      # 모델 생성 및 학습
├── cnn_text_recognition.py     # 모델 불러와 영상에서 글자 인식 (main)
├── collect_image.py            # dataset 직접 수집
├── resize_image.py             # dataset 한번에 리사이즈
└── text_CNN.h5                 # weight 파일
``` 


![슬라이드4](https://user-images.githubusercontent.com/54545026/123627808-61cf7780-d84d-11eb-8802-cc40cbdae314.PNG)
![슬라이드13](https://user-images.githubusercontent.com/54545026/123627814-65fb9500-d84d-11eb-8da9-4ca4511a0036.PNG)

### label
![슬라이드14](https://user-images.githubusercontent.com/54545026/123627957-8fb4bc00-d84d-11eb-96b0-d29ac5c60040.PNG)
