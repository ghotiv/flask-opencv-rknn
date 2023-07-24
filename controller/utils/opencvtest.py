import cv2
import numpy as np
from PIL import Image

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=1000, detectShadows=False)

train = 0
trainLabels = 0

flag = 0

def getPerson(image, opt=1):

    # get the front mask
    mask = fgbg.apply(image)

    # eliminate the noise
    line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5), (-1, -1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, line)

    # find the max area contours
    out, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if area < 4500:
            continue
        rect = cv2.minAreaRect(contours[c])
        cv2.ellipse(image, rect, (0, 255, 0), 2, 8)
        cv2.circle(image, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
    return image, mask


# class OpencvTest(object):
#     def __init__(image):
#         # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         # # 进行高斯模糊操作
#         # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         # # 执行边缘检测
#         # edged = cv2.Canny(blurred, 50, 200, 255)
#         # cv2.imwrite("new_image.png", edged)

#         # new_image = cv2.imread("new_image.png")

#         return image

def OpencvTest1(image):
    # new_image = []
    # 将输入转换为灰度图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行高斯模糊操作
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 执行边缘检测
    edged = cv2.Canny(blurred, 100, 200, 50)
    
    cv2.imwrite("new_image.png", edged)

    new_image = cv2.imread("new_image.png")

    return new_image
    
def OpencvTest2(image):
    # 转换到 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #设定蓝色的阈值。确定要追踪的颜色为白色。
    lower_blue = np.array([0,0,221])
    upper_blue = np.array([180,22,255])

    #根据阈值构建掩模，构建黑白图
    #hsv:原图
    #lower_blue:图像中低于这个lower_blue的值，图像值变为0,即黑色
    #upper_blue:图像中高于这个upper_blue的值，图像值变为0
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 对原图像和掩模进行位运算
    res = cv2.bitwise_and(image, image, mask=mask)

    return image

def load_data():
    global trainLabels 
    global train 
    global flag
    if (flag == 0):
        flag = 1
        with np.load('./data.npz') as data:
            print(data.files)
            train = data['train']
            trainLabels = data['train_labels']

# 定义平移translate函数
def translate(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])

    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=(255,255,255))

    # 返回转换后的图像
    return shifted

def number_recognition(image):
    global trainLabels 
    global train 
    #读取待识别特征值
    row = 20  #特征图像的行数
    col = 20  #特征图像的列数
    # 变成多少分辨率的方形图
    # img_width_height=20

    # tmp_img = make_square(image, fill_color=(255, 255, 255, 0))

    # # 读取原图片
    # h, w = tmp_img.shape[0], tmp_img.shape[1]

    # rate=img_width_height/h
    # img_processing = cv2.resize(tmp_img, (0, 0), fx=rate, fy=rate, interpolation=cv2.INTER_NEAREST)
    img_processing = cv2.resize(image,(row,col),interpolation=cv2.INTER_CUBIC)

    # img_processing = cv2.threshold(img_processing,127,255,cv2.THRESH_BINARY)

    ret1, th1 = cv2.threshold(img_processing, 127, 255, cv2.THRESH_BINARY)

    th2 = translate(th1, 2, 2)

    cv2.imwrite("new_image.png", th2)

    new_image = cv2.imread("new_image.png",0)
    # o = cv2.imread('./number_image-bak1/5-3.png',0)
    
    # of = np.zeros((round(row),round(col)))  #存储待识别图像特征值

    # for nr in range(0,row):
    #     for nc in range(0,col):
    #         if new_image[nr,nc].all() == 255:
    #             of[int(nr),int(nc)]+=1
    
    test = new_image.reshape(-1,round(row)*round(col)).astype(np.float32)
    # test = x[:,50:100].reshape(-1,400).astype(np.float32)
    #调用函数识别图像
    knn = cv2.ml.KNearest_create()
    knn.train(train,cv2.ml.ROW_SAMPLE,trainLabels)
    ret,result,neighbours,dist = knn.findNearest(test,k=7)
    # print("当前的数字可能为:",result)

    return result

def OpencvTest3(image):
    global trainLabels 
    global train 
    load_data()
    # 将输入转换为灰度图片，并去噪声
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (9, 9),0)

    # # threshold(imag, result, 30, 200.0, CV_THRESH_BINARY)
    # # ret, th1 = cv2.threshold(image,100,255,cv2.THRESH_BINARY)

    # # global thresholding
    # ret1, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # # Otsu's thresholding
    # ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
    # # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(image, (5, 5), 0)
    # ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)

    # #提取图像的梯度

    # gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    # gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    # gradient = cv2.subtract(gradX, gradY)
    # gradient = cv2.convertScaleAbs(gradient)

    # # 执行边缘检测
    # edged = cv2.Canny(gradient, 150, 250, 50)

    # # blurred = cv2.GaussianBlur(gradient, (9, 9),0)
    # # (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # img = cv2.imread('/home/pi/opencv/digits.png')

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    ## 阈值分割
    ret,thresh = cv2.threshold(gray, 100, 255, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5, 5))     
    dilated = cv2.dilate(thresh, kernel, iterations = 1)

    # ## 轮廓提取
    # image1, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # 检测所有轮廓，轮廓间建立外层、内层的等级关系，并且保存轮廓上所有点
    # image1, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 检测所有轮廓，但各轮廓之间彼此独立，不建立等级关系，并且仅保存轮廓上拐点信息
    image1, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    for i in range(len(contours)):
        cnt = contours[i]

        # 绘制轮廓
        # cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(cnt)
        max_len = max(w, h)
        area = max_len * max_len

        M = cv2.moments(cnt)#计算第一条轮廓的矩
        #这两行是计算中心点坐标
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # (x1,y1), radius = cv2.minEnclosingCircle(cnt)

        # center = (int(x1),int(y1))

        # 滤除其他异常的轮廓区域
        if area > 7000 and area < 38000 and abs(w - h) < 100:
            # roi = cv2.Rect(x, y, x + w, y + h)

            cropped = image[y:y + max_len, x:x + max_len]
            number = number_recognition(cropped)

            # 绘制外接矩形
            cv2.rectangle(image, (x , y), (x + max_len, y + max_len), (0, 255, 0), 3)
            # cv2.circle(image, (cx, cy), 10, (255, 0, 0), 2)
            # cv2.circle(image,(447,63), 63, (0,0,255), -1)
            # temp = cv2.srcImg(x, y, x + w, y + h)

            
            # 标记面积
            cv2.putText(image, str(number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

    return image





