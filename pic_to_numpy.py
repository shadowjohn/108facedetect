# -*- coding: utf-8 -*-
import php
import sys
import os
from skimage import io
import dlib
import numpy
my = php.kit()

# 人臉68特徵點模型路徑
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# 載入人臉檢測器
detector = dlib.get_frontal_face_detector()
# 載入人臉特徵點檢測器
sp = dlib.shape_predictor(predictor_path)
# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

fp = my.glob("rec\\pic\\*.jpg")
np_dir = "rec\\numpy"

# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子
descriptors = []
step = 0
totals = len(fp)
for f in fp:
  print("Run ... " + str(step) + " / " + str((totals-1)) )
  step = step + 1
  base = os.path.basename(f)
  mn = my.mainname(f)
  op_txt = np_dir+"\\"+mn
  if my.is_file(op_txt+".npy") == True:
    continue
  img = io.imread(f)
  # 1.人臉偵測
  dets = detector(img, 1)
  for k, d in enumerate(dets):
    # 2.特徵點偵測
    shape = sp(img, d)
    # 3.取得描述子，128維特徵向量
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    # 轉換numpy array格式
    v = numpy.array(face_descriptor)
    #descriptors.append(v)
    #print(descriptors)
    # https://ithelp.ithome.com.tw/articles/10196167
    numpy.save(op_txt,v)