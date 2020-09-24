import cv2


def show(image):
    cv2.imshow('', image)
    cv2.waitKey()
    cv2.destroyAllWindows()


path = '/media/david/A/Datasets/PlayHippo/masked_images_cleaned/huawei_ale_l21/bear/IMG_19700105_023522.jpg'

image = cv2.imread(path)
image = cv2.resize(image, (512, 512))

show(image)

ksize = (0, 0)

borders = [
cv2.BORDER_CONSTANT,
cv2.BORDER_REPLICATE,
cv2.BORDER_REFLECT,
cv2.BORDER_WRAP,
cv2.BORDER_REFLECT_101,
cv2.BORDER_TRANSPARENT,
cv2.BORDER_REFLECT101,
cv2.BORDER_DEFAULT,
cv2.BORDER_ISOLATED
]

for border in borders:
    show(cv2.GaussianBlur(image, ksize, border))

exit()
blured = cv2.GaussianBlur(image, ksize, cv2.BORDER_REFLECT)
# blured = cv2.GaussianBlur(blured, ksize, cv2.BORDER_DEFAULT)
# blured = cv2.GaussianBlur(blured, ksize, cv2.BORDER_DEFAULT)

show(blured)
