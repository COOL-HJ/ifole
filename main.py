# 指定人脸匹配相似佛脸
import os, dlib, glob, numpy
import numpy as np

from skimage import io
from PIL import Image

# 人脸关键点检测器
predictor_path = "data/shape_predictor_68_face_landmarks.dat"
# 人脸识别模型、提取特征值
face_rec_model_path = "data/dlib_face_recognition_resnet_model_v1.dat"

# 加载模型
# detector = dlib.get_frontal_face_detector()
detector = dlib.simple_object_detector("data/detector_buddhism.svm")
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

candidate = np.load("data/candidate_buddhism.npy")  # 训练集人物名字
descriptors = np.load("data/descriptors_buddhism.npy")  # 训练集人物特征列表

try:
    ##    test_path=input('请输入要检测的图片的路径（记得加后缀哦）:')
    imgname = "faces/image.jpg"
    img = io.imread(r"faces/image.jpg")
    imgshow = Image.open(imgname)
    imgshow.show()
    dets = detector(img, 1)
except:
    print('输入路径有误，请检查！')

dist = []
for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor)
    for i in descriptors:  # 计算距离
        dist_ = numpy.linalg.norm(i - d_test)
        dist.append(dist_)

# 训练集人物和距离组成一个字典
c_d = dict(zip(candidate, dist))
cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
print("识别到的人物最有可能是: ", cd_sorted[0][0], "（结果越小表明越相似）")
print(cd_sorted)
print(dist)
filename = "faces/buddhism/" + cd_sorted[0][0] + ".jpg"
print(filename)
img = Image.open(filename)
img.show()