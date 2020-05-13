import cv2


for i in range(1, 101):
	bw = cv2.imread("../../data/training/training/groundtruth/satImage_%03d.png" % i)[:, :, 0]
	_, bw2 = cv2.threshold(bw, 10, 255, cv2.THRESH_BINARY)
	bw2 = cv2.Canny(bw2, 100, 200)
	cv2.imwrite("edges/atImage_%03d.png" % i, bw2)