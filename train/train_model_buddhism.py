# 保存佛像训练集特征列表以及名字
import os, dlib, glob, numpy
import numpy as np

from skimage import io

# 佛脸关键点检测器
predictor_path = "../data/shape_predictor_68_face_landmarks.dat"
# 佛脸识别模型、提取特征值
face_rec_model_path = "../data/dlib_face_recognition_resnet_model_v1.dat"
# 训练图像文件夹
faces_folder_path = '../faces/buddhism'

# 加载模型
# detector = dlib.get_frontal_face_detector()
detector = dlib.simple_object_detector("detector_foxiang.svm")
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

candidate = []  # 存放训练集人物名字
descriptors = []  # 存放训练集人物特征列表

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("正在处理: {}".format(f))
    img = io.imread(f)
    candidate.append(f.split('/')[-1].split('.')[0])
    # 人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        # 提取特征
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor)
        descriptors.append(v)

print('识别训练完毕！')
np.save("../data/candidate_buddhism.npy", candidate)
np.save("../data/descriptors_buddhism.npy", descriptors)
