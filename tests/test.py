import dlib
from imageio import imread
import glob
import time
import numpy as np

win = dlib.image_window()
detector = dlib.get_frontal_face_detector()
predictor_path = 'data/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_rec_model_path = 'data/dlib_face_recognition_resnet_model_v1.dat'
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
paths = glob.glob('faces/3.jpg')
labeled = glob.glob('labeled/*.jpg')
labeled_data = {}
unlabeled = glob.glob('unlabeled/*/*.jpg')


# test
def distance(a, b):
    # d = 0
    # for i in range(len(a)):
    # 	d += (a[i] - b[i]) * (a[i] - b[i])
    # return np.sqrt(d)
    return np.linalg.norm(np.array(a) - np.array(b), ord=2)


# 读取标注图片并保存对应的128向量
for path in labeled:
    img = imread(path)
    name = path.split('\\')[1].rstrip('.jpg')
    dets = detector(img, 1)
    # 这里假设每张图只有一个人脸
    shape = predictor(img, dets[0])
    face_vector = facerec.compute_face_descriptor(img, shape)
    labeled_data[name] = face_vector

# 读取未标注图片，并和标注图片进行对比
for path in unlabeled:
    img = imread(path)
    name = path.split('\\')[2].rstrip('.jpg')
    dets = detector(img, 1)
    if dets is not None:
        # 这里假设每张图只有一个人脸
        shape = predictor(img, dets[0])
        face_vector = facerec.compute_face_descriptor(img, shape)
        matches = []
        for key, value in labeled_data.items():
            d = distance(face_vector, value)
            if d < 0.6:
                matches.append(key + ' %.4f' % d)
                # win.clear_overlay()
                # win.set_image(img)
        print('{}---{}'.format(name, ';'.join(matches)))


# for path in paths:
#     img = imread(path)
#     win.clear_overlay()
#     win.set_image(img)
#
#     # 1 表示将图片放大一倍，便于检测到更多人脸
#     dets = detector(img, 1)
#     print('检测到了 %d 个人脸' % len(dets))
#     for i, d in enumerate(dets):
#         print('- %d: Left %d Top %d Right %d Bottom %d' % (i, d.left(), d.top(), d.right(), d.bottom()))
#         shape = predictor(img, d)
#         # 第 0 个点和第 1 个点的坐标
#         print('Part 0: {}, Part 1: {}'.format(shape.part(0), shape.part(1)))
#         win.add_overlay(shape)
#
#     win.add_overlay(dets)
#     # dlib.hit_enter_to_continue()
#     time.sleep(2)