# -*- coding: utf-8 -*-
import dlib
import cv2
import imutils
import php
import os
import numpy
import time
my = php.kit()
# 比對人臉解算資料夾名稱
faces_numpy_folder_path = "./rec/numpy"

# 人臉辨識模型路徑
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# 載入人臉辨識檢測器
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# 比對人臉描述子列表
descriptors = []

# 比對人臉名稱列表
candidate = []

# 針對比對資料夾裡每張圖片做比對:
# 1.人臉偵測
# 2.特徵點偵測
# 3.取得描述子
for f in my.glob(faces_numpy_folder_path+"\\*.npy"):
  base = os.path.basename(f)
  # 依序取得圖片檔案人名
  candidate.append(os.path.splitext(base)[ 0])
  # from : https://ithelp.ithome.com.tw/articles/10196167
  v = numpy.load(f)
  descriptors.append(v)

#選擇第一隻攝影機
cap = cv2.VideoCapture( 0)
#調整預設影像大小，預設值很大，很吃效能
cap.set(cv2. CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2. CAP_PROP_FRAME_HEIGHT, 240)

#取得預設的臉部偵測器
detector = dlib.get_frontal_face_detector()
#根據shape_predictor方法載入68個特徵點模型，此方法為人臉表情識別的偵測器
predictor = dlib.shape_predictor( 'shape_predictor_68_face_landmarks.dat')
#當攝影機打開時，對每個frame進行偵測
step_while = 0;
x1=0
last_rec_name=""
while(cap.isOpened()):
  #time.sleep(0.5)
  #讀出frame資訊
  ret, frame = cap.read()  
  step_while=step_while+1
  #取出偵測的結果
  print(step_while)
  if step_while>60:
    x1=0
    last_rec_name = ""
    step_while = 0 
  if step_while % 30==0:
    dist = []
    #偵測人臉
    face_rects, scores, idx = detector.run(frame, 0)    
    for i, d in enumerate(face_rects):
      x1 = d.left()
      y1 = d.top()
      x2 = d.right()
      y2 = d.bottom()
      #text = " %2.2f ( %d )" % (scores[i], idx[i])
  
      #繪製出偵測人臉的矩形範圍
      cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
  
      #標上人臉偵測分數與人臉方向子偵測器編號
      #cv2.putText(frame, text, (x1, y1), cv2. FONT_HERSHEY_DUPLEX,
      #0.7, ( 255, 255, 255), 1, cv2. LINE_AA)
  
      #給68特徵點辨識取得一個轉換顏色的frame
      landmarks_frame = cv2.cvtColor(frame, cv2. COLOR_BGR2RGB)
  
      #找出特徵點位置
      shape = predictor(landmarks_frame, d)
      face_descriptor = facerec.compute_face_descriptor(frame, shape)
      d_test = numpy.array(face_descriptor)
      # 計算歐式距離
      for j in descriptors:
        dist_ = numpy.linalg.norm(j - d_test)
        dist.append(dist_)    
                      
        #繪製68個特徵點
        #for i in range( 68):
        #  cv2.circle(frame,(shape.part(i).x,shape.part(i).y), 3,( 0, 0, 255), 2)
        #  cv2.putText(frame, str(i),(shape.part(i).x,shape.part(i).y),cv2. FONT_HERSHEY_COMPLEX, 0.5,( 255, 0, 0), 1)
        #輸出到畫面
        # 將比對人名和比對出來的歐式距離組成一個dict
        c_d = dict( zip(candidate,dist))
        
        # 根據歐式距離由小到大排序
        #cd_sorted = sorted(c_d.iteritems(), key = lambda d:d[ 1])
        cd_sorted = sorted(c_d.items(), key=lambda kv: kv[1])
        # 取得最短距離就為辨識出的人名
        #print(cd_sorted)
        if cd_sorted[0][1]<0.4:
          rec_name = cd_sorted[0][0]
          m = my.explode("#",rec_name)
          last_rec_name = m[0]
          #首字大寫
          last_rec_name = my.strtolower(last_rec_name)
          last_rec_name = last_rec_name.capitalize()
          step_while = 0          
          # 將辨識出的人名印到圖片上面
          cv2.putText(frame, last_rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2. LINE_AA)        
          print(m[0])
  if x1!=0:
    cv2.rectangle(frame, (x1, y1), (x2, y2), ( 0, 255, 0), 4, cv2. LINE_AA)
  if last_rec_name!="":    
    cv2.putText(frame, last_rec_name, (x1, y1), cv2. FONT_HERSHEY_SIMPLEX , 1, ( 255, 255, 255), 2, cv2. LINE_AA)          
  cv2.imshow( "Face Detection", frame)

  #如果按下ESC键，就退出
  if cv2.waitKey( 10) == 27:
     break
#釋放記憶體
cap.release()
#關閉所有視窗
cv2.destroyAllWindows()