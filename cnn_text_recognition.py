import cv2 as cv
import numpy as np
from keras.models import load_model
from PIL import Image

cap = cv.VideoCapture(0)
Nframe = 0  # frame 수

color_flag = 'No color to detect.'

while cap.isOpened():
    ret, img = cap.read()

    if ret:  # 비디오 프레임  을 읽기 성공했으면 진행
        img = cv.resize(img, (500, 300))
        img_origin = img[:, :].copy()
    else:
        break

    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Binarization
    ret, binary = cv.threshold(gray, 100, 255, 0)

    # morph gradient
    kernel1 = np.ones((2, 2), np.uint8)
    erosion = cv.erode(gray, kernel1, iterations=1)
    dilation = cv.dilate(gray, kernel1, iterations=1)

    morph = dilation - erosion

    # cv.imshow("gray", gray)            # Grayscale
    # cv.imshow("erosion", erosion)      # erosion (침식 연산 이미지)
    # cv.imshow("dilation", dilation)    # dilation (팽창 연산 이미지)
    cv.imshow("morph", morph)            # 윤곽선 이미지 (팽참-침식)

    # long line remove(HoughLinesP)
    # edges = cv.Canny(morph, 50, 200)
    # lines = cv.HoughLinesP(edges, 1, np.pi / 180., 80, minLineLength=30, maxLineGap=2)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv.line(morph, (x1, y1), (x2, y2), (0, 0, 0), 4)  # 긴 직선 지우기 (검정으로 변환)
    #
    # cv.imshow('edges', edges)               # canny edge 이미지
    # cv.imshow('remove_long_line', morph)    # 직선 지운 윤곽선 이미지


    # contours
    ret, img_binary = cv.threshold(morph, 30, 255, 0)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv.drawContours(img, [cnt], 0, (0, 255, 0), 1)       # 경계 그룹 선으로 표현


    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 100 and h > 100 and w < 300:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            text = binary[y:(y + h), x:(x + w), ]
            cv.imshow("text", text)

            text = cv.resize(text, (28, 28))
            text = np.array(text, dtype=np.int32).reshape(28, 28, 1)

            imgs = np.zeros(1 * 28 * 28 * 1, dtype=np.int32).reshape(1, 28, 28, 1)
            imgs[0, :, :, :] = text

            loaded_model = load_model('text_CNN.h5')
            prediction = loaded_model.predict(imgs)

            print("label: ",str(np.argmax(prediction)))
            if np.argmax(prediction) == 0:
                cv.putText(img, "A", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif np.argmax(prediction) == 1:
                cv.putText(img, "B", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif np.argmax(prediction) == 2:
                cv.putText(img, "C", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif np.argmax(prediction) == 3:
                cv.putText(img, "D", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif np.argmax(prediction) == 4:
                cv.putText(img, "E", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif np.argmax(prediction) == 5:
                cv.putText(img, "W", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif np.argmax(prediction) == 6:
                cv.putText(img, "S", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)
            elif np.argmax(prediction) == 7:
                cv.putText(img, "N", (30,60),cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv.LINE_AA)


    cv.imshow("con2", img)
    cv.imshow("bin", binary)
    cv.imshow("bin", img_binary)

    if cv.waitKey(25) == 27:
        break

print("Number of Frame: ", Nframe)  # 영상의 frame 수 출

cap.release()
cv.destroyAllWindows()