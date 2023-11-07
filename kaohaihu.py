import cv2
import numpy as np
cascade = cv2.CascadeClassifier("/home/serveradmin/Desktop/cascade/data/haarcascades/haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
img = cv2.imread("/home/serveradmin/Desktop/GAZO/SAKABANN.png")
shot = 0
if not cap.isOpened():
    print("CameraDeath")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERRoR")
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hc,wc = frame.shape[:2]
    lists=cascade.detectMultiScale(frame_gray,minSize=(150,150))
    #img_warped = frame
    for (x,y,w,h) in lists:
        ht = 279
        wt = 795
        Sx = abs(round(abs(w)/7.95)/100)#X/Y拡大率
        Sy =abs(round(((279/795)*abs(w))/2.79)/100)
        dx = x#移動
        dy = y+round((abs(h)/3))-round((279/1590)*(abs(w)))
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255), thickness=2)
        M = np.array([[Sx ,0 ,dx], [0 ,Sy ,dy]], dtype=float)
        frame = cv2.warpAffine(img, M, (wc,hc),frame,borderMode= cv2.BORDER_TRANSPARENT)
    cv2.imshow('video image', frame)
    cv2.setWindowProperty('video image',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()