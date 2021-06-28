# CNN_text_classification
A program that uses CNN to detect alphabet cards and recognize letters in videos. 
<br>CNN 모델을 사용하여 영상에서 알파벳 카드를 감지하고 글자를 인식하는 프로그램입니다. 
<br>

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
