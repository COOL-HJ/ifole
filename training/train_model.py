import os, dlib, glob, numpy
import numpy as np
import time

from skimage import io

# 人脸关键点检测器
predictor_path = "data/shape_predictor_68_face_landmarks.dat"
# 人脸识别模型、提取特征值
face_rec_model_path = "data/dlib_face_recognition_resnet_model_v1.dat"
# 训练图像文件夹
faces_folder_path = 'faces/lfw/*'

# 加载模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

candidate = []  # 存放训练集人物名字
descriptors = []  # 存放训练集人物特征列表

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("正在处理: {}".format(f))
    img = io.imread(f)
    candidate.append(f.split('\\')[-2].split('.')[0])
    # 人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        # 提取特征
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor)
        descriptors.append(v)

print('识别训练完毕！')

candidate = np.array(candidate)
np.save('data/candidate.npy', candidate)
descriptors = np.array(descriptors)
np.save('data/descriptors.npy', descriptors)
print('训练结果已保存！')
