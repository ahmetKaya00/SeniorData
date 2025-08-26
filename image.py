import cv2


image = cv2.imread("example.png")

print("Görüntü Boyutu:", image.shape)

if image is None:
    print("Hata: Görüntü dosyası bulunamadı!")
    exit()

cv2.imshow("Goruntu",image)

cv2.waitKey(0)
cv2.destroyAllWindows()

image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_example.png",image_gray)

new_width = 400
new_height = int(image.shape[0] * (new_width / image.shape[1]))

resized_image = cv2.resize(image,(new_width,new_height))
print("Görüntü Boyutu:", resized_image.shape)

cv2.imshow("Goruntu",image)

cv2.waitKey(0)
cv2.destroyAllWindows()

rotate_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Goruntu",rotate_90)

cv2.waitKey(0)
cv2.destroyAllWindows()

cropping_image = image[50:250,50:250]
cv2.imshow("Goruntu",cropping_image)

cv2.waitKey(0)
cv2.destroyAllWindows()