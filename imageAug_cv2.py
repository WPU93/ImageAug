import cv2
import os
import numpy as np

def flip(srcImg,param):
    '''
    :param srcImg:
    :param param: param = 0：垂直翻转(沿x轴)，param > 0: 水平翻转(沿y轴)，param < 0: 水平垂直翻转
    :return:
    '''
    dstImg = cv2.flip(srcImg,param)
    return dstImg

def rotation(srcImg,param):
    '''
    :param srcImg:
    :param param:顺时针param%4个90度
    :return:
    '''
    dstImg = srcImg.copy()
    if param %4 == 0:
        dstImg = srcImg
    elif param %4 == 1:
        srcImg = cv2.transpose(srcImg)
        dstImg = cv2.flip(srcImg, 1)
    elif param %4 == 2:
        srcImg = cv2.flip(srcImg, 0)
        dstImg = cv2.flip(srcImg, 1)
    elif param %4 == 3:
        srcImg = cv2.transpose(srcImg)
        dstImg = cv2.flip(srcImg, 0)

    return dstImg

def rotation_angle(srcImg,angle,ratio = 1):
    '''
    :param srcImg:
    :param angle: 旋转角度，大于0逆时针，（采用仿射变换实现
    :param ratio: 缩放比例
    :return:
    '''
    dstImg = srcImg.copy()
    cols, rows,_ = srcImg.shape
    # print(cols,rows)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, ratio)
    dstImg = cv2.warpAffine(srcImg, M, (cols, rows))
    return dstImg
def mixed_img(img1,img2,w1,w2):
    '''
    :param img1:
    :param img2:
    :param w1: 图片1的混合权重
    :param w2: 图片2的混合权重
    :return:
    '''
    dstImg = cv2.addWeighted(img1, w1, img2,w2, 0)
    return dstImg

def masked_img(img,mask):
    pass
    return dstImg

def brightness_alpha_beta(srcImg,alpha,beta):
    '''增益与偏置值法调节对比度亮度
    :param srcImg:
    :param alpha: 对比度
    :param beta:亮度
    :return:
    '''
    dstImg = np.uint8(np.clip((alpha * srcImg + beta), 0, 255))
    return dstImg

def equalize_hist(img):
    '''
    直方图均衡技术,对比度较低的图像，并增加图像相对高低的对比度，以便在阴影中产生细微的差异
    :param img:
    :return:
    '''
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb_img)
    channels[0] = cv2.equalizeHist(channels[0])
    ycrcb_img = cv2.merge(channels)
    dstImg = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCR_CB2BGR)
    return dstImg

def equalize_ada(img):
    '''
    自适应均衡技术,对比度较低的图像，并增加图像相对高低的对比度，以便在阴影中产生细微的差异
    :param img:
    :return:
    '''
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    channels[0] = clahe.apply(channels[0])
    ycrcb_img = cv2.merge(channels)
    dstImg = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCR_CB2BGR)
    return dstImg

def add_salt_pepper(src,percetage):
    '''
    椒盐噪声
    :param src: 原始图片
    :param percetage:
    :return:
    '''
    SP_NoiseImg=src.copy()
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randR=np.random.randint(0,src.shape[0]-1)
        randG=np.random.randint(0,src.shape[1]-1)
        randB=np.random.randint(0,3)
        if np.random.randint(0,1)==0:
            SP_NoiseImg[randR,randG,randB]=0
        else:
            SP_NoiseImg[randR,randG,randB]=255
    return SP_NoiseImg

def add_gaussian_noise(image,percetage):
    '''
    高斯噪声
    :param image:
    :param percetage:
    :return:
    '''
    G_Noiseimg = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,h)
        temp_y = np.random.randint(0,w)
        G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
    return G_Noiseimg

def blur(srcImg,type="gaussian",param=None):
    '''
    :param srcImg:
    :param type:模糊类型（高斯滤波，方框滤波，中值滤波，双边滤波）
    :param param:
    :return:
    '''
    dstImg = srcImg.copy()
    if type == "gaussian":
        dstImg = cv2.blur(srcImg,(5,5))
    elif type == "box":
        dstImg = cv2.GaussianBlur(srcImg,(5,5),0)
    elif type == "median":
        dstImg = cv2.medianBlur(srcImg,5)
    elif type == "bilateral":
        dstImg = cv2.bilateralFilter(srcImg,9,75,75)

    return dstImg

def sharpen(srcImg):
    """
    高通滤波器锐化
    :param srcImg:
    :return:
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化卷积核
    dstImg = cv2.filter2D(srcImg,-1,kernel)
    return dstImg
if __name__=="__main__":
    srcImg = cv2.imread("resources\\8.jpg")
    srcImg2 = cv2.imread("resources\\9.jpg")
    dstImg = mixed_img(srcImg,srcImg2,0.7,0.3)
    tmp = np.hstack((srcImg,srcImg2))
    display = np.hstack((tmp, dstImg))
    cv2.imshow("display",display)
    cv2.imwrite("resources\\mix.jpg",display)
    cv2.waitKey(0)
