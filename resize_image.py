from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import glob

# 디렉토리 경로 설정
img_path ='Data/test_dataset/N/'
images = glob.iglob(img_path + "*.jpg")

# 파일마다 모두 28x28 사이즈로 바꾸어 저장
target_size = (28, 28)

for img in images:
    old_img = Image.open(img)
    new_img = old_img.resize(target_size, Image.ANTIALIAS)
    new_img.save(img, "JPEG")

print("resize 완료!")