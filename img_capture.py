import cv2
import numpy as np

cap=cv2.VideoCapture(0)
pic_no=0
total_pic=1200
flag_capturing= False
path='D:/Datascience/Sign-Language-Alphabet-recognizer/mydata/'#Save for each alphabet and specify the path as /dataset/A,/dataset/B, etc..
while(cap.isOpened()):
    rval,frame=cap.read()
    frame=cv2.flip(frame,1)

    cv2.rectangle(frame,(300,300),(100,100),(0,255,0),0)

    cv2.imshow("Image",frame)
    crop_img=frame[100:300,100:300]
    keypress=cv2.waitKey(1)

    if flag_capturing:
        pic_no+=1
        save_img=cv2.resize(crop_img,(50,50))
        save_img=np.array(save_img)
        # if keypress==ord('s'):
        cv2.imwrite(path+"/"+str(pic_no)+".jpg",save_img)
        print("Saving image: "+str(pic_no))

    

    if pic_no==total_pic:
        flag_capturing=False
        break

    if keypress==ord("q"):
        break
    elif keypress ==ord("c"):
        flag_capturing=True

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)