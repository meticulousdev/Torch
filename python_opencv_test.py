import cv2

image = cv2.imread("python.png", cv2.IMREAD_COLOR)
cv2.imshow("python", image)
cv2.waitKey(0)
cv2.destroyAllWindows()