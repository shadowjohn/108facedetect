# 108facedetect
<h2>人臉辨識練習小程式</h2>
版本：V0.1<br>
<br>
用途：可以作為門禁、打卡功能快速開發<br>
<br>
縮圖參考：<br>
  <img src="screenshot/screenshot1.png">
<br>
相依程式：<br>
　　python 3.6.4 x86 (32bit)<br>
　　pip install scipy<br>
　　pip install dlib<br>
　　pip install numpy<br>
　　pip install opencv-python<br>
　　pip install scikit-image<br>
　　pip install matplotlib<br>
　　pip install imutils<br>
　　Webcam 一組<br>
<br>
第三方下載：<br>
　　68特徵：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2<br>
　　人臉模型：http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2<br>
<br>
使用方式：<br>
　　1、先收集所有屁孩的大頭照或生活照，正面、臉清晰照片，檔名為屁孩的名字，存至 rec/pic，如 rec/pic/john.jpg，同一個人越多張越好，如 rec/pic/john#1.jpg rec/pic/john#2.jpg<br>
　　2、將照片作出特徵檔，執行 pic_to_numpy.py：python pic_to_numpy.py<br>
　　3、啟動 Webcam：python cam.py<br>
<br>
程式說明：<br>
　　rec/pic 目錄，放人臉，一個人一張，如 john.jpg，同一個jpg只能有一個人，同一個人請多檔，如john#1.jpg john#2.jpg<br>
　　rec/numpy 目錄，透過 pic_to_numpy.py 執行後，會把 rec/pic 目錄下所有人作出特徵檔 rec/numpy/人名.npy 檔案<br>
　　pic_to_numpy.py 把 rec/pic 目錄下的所有照片，算出特徵檔，然後會產生到 rec/numpy 下，可以加快辨識速度。<br>
　　cam.py 啟動Webcam，人臉會被自動框出，每30個影格判斷一次該影格的人是誰。<br>
<br>
參考資料：<br>
　　1、https://tpu.thinkpower.com.tw/tpu/articleDetails/950<br>　　