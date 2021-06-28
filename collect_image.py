import cv2 as cv

# 웹캠 연결
cap = cv.VideoCapture(0)
N_frame = 0

while cap.isOpened():
    N_frame += 1
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame.")
        break

    # Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Binarization
    ret, binary = cv.threshold(gray, 100, 255, 0)

    # 저장할 이미지 사이즈 지정
    # resize하면 해당 하이즈로 이미지 변형 (왜곡됨) dsize=(x, y)
    # img = cv.resize(frame, dsize=(100, 400))

    # resize하지 않고 슬라이싱해서 복사 [y, x]
    img = binary[:400, :400]

    # 지정된 디렉토리에 프레임마다 프레임 번호를 파일명으로 저장
    cv.imwrite("Data/test_dataset/N/"+str(N_frame)+".jpg", img)
    cv.imshow('video', img)
    print(N_frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()