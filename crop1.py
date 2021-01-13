import cv2,os
import numpy as np
from matplotlib import pyplot

pathname='img/juice.jpg'
os.makedirs('result/' , exist_ok=True)
image = cv2.imread(pathname)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 用Sobel算子計算x，y方向上的梯度，之後在x方向上減去y方向上的梯度，通過這個減法，我們留下具有高水平梯度和低垂直梯度的圖像區域
gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# 去除圖像上的噪聲。首先使用低通濾潑器平滑圖像（9 x 9內核）,這將有助於平滑圖像中的高頻噪聲。低通濾波器的目標是降低圖像的變化率。如將每個像素替換為該像素周圍像素的均值。這樣就可以平滑並替代那些強度變化明顯的區域
# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9)) 
height = blurred.shape[0]		
width = blurred.shape[1]		
dstHeight = int(height * 0.5)		
dstWidth = int(width * 0.5)		

# blurred1 = cv2.resize(blurred, (dstWidth,dstHeight))

(_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
# cv2.imshow("Image",blurred1)
# cv2.waitKey(0)

# 我們要用白色填充黑色的空餘，使得後面的程序更容易識別，這需要做一些形態學方面的操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 圖像上還有一些小的白色斑點，這會干擾之後的輪廓的檢測，要把它們去掉。分別執行4次形態學腐蝕與膨脹。
# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4) 

# 找出區域的輪廓。cv2.findContours()函數第一個參數是要檢索的圖片，必須是為二值圖，即黑白的（不是灰度圖），所以讀取的圖像要先轉成灰度的，再轉成二值圖
# cv2.minAreaRect()函數:
# 主要求得包含點集最小面積的矩形，這個矩形是可以有偏轉角度的，可以與圖像的邊界不平行。
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int0(cv2.boxPoints(rect))

# draw a bounding box arounded the detected barcode and display the image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
image = cv2.resize(image, (dstWidth,dstHeight))
cv2.imshow("Image", image)
cv2.imwrite("result/contoursImage2.jpg", image)
cv2.waitKey(0)

# 裁剪。box裡保存的是綠色矩形區域四個頂點的坐標。我將按下圖紅色矩形所示裁剪昆蟲圖像。找出四個頂點的x，y坐標的最大最小值。新圖像的高=maxY-minY，寬=maxX-minX
Xs = [i[0] for i in box]
Ys = [i[1] for i in box]
x1 = min(Xs)
x2 = max(Xs)
y1 = min(Ys)
y2 = max(Ys)
hight = y2 - y1
width = x2 - x1
length = hight if hight>width else width

image = cv2.imread(pathname)
cropImg = image[y1:y1+length, x1:x1+length]
cv2.imwrite('result/output.jpg', cropImg)

################################################################

# 取得紅色方框的旋轉角度
angle = rect[2]
if angle < -45:
  angle = 90 + angle

# 以影像中心為旋轉軸心
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# 計算旋轉矩陣
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# 旋轉圖片
img_debug = image.copy()
rotated = cv2.warpAffine(img_debug, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
img_final = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

# 除錯用的圖形
pyplot.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
pyplot.show()

# 旋轉紅色方框座標
pts = np.int0(cv2.transform(np.array([box]), M))[0]

# 計算旋轉後的紅色方框範圍
y_min = min(pts[0][0], pts[1][0], pts[2][0], pts[3][0])
y_max = max(pts[0][0], pts[1][0], pts[2][0], pts[3][0])
x_min = min(pts[0][1], pts[1][1], pts[2][1], pts[3][1])
x_max = max(pts[0][1], pts[1][1], pts[2][1], pts[3][1])

# 裁切影像
img_crop = rotated[x_min:x_max, y_min:y_max]
img_final = img_final[x_min:x_max, y_min:y_max]

# 除錯用的圖形
pyplot.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
pyplot.show()