import cv2
import numpy as np

def nothing(x):
    pass

image_x,image_y=64,64

from keras.models import load_model
classifier=load_model('SLRmodel.h5')

def predictor():
    import numpy as np
    from keras.preprocessing import image
    test_image=image.load_img('1.png',target_size=(64,64))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image, axis=0)
    result=classifier.predict(test_image)

    if result[0][0] == 1:
        return 'A'
    elif result[0][1] == 1:
              return 'B'
    elif result[0][2] == 1:
              return 'C'
    elif result[0][3] == 1:
              return 'D'
    elif result[0][4] == 1:
              return 'E'
    elif result[0][5] == 1:
              return 'F'
    elif result[0][6] == 1:
              return 'G'
    elif result[0][7] == 1:
              return 'H'
    elif result[0][8] == 1:
              return 'I'
    elif result[0][9] == 1:
              return 'J'
    elif result[0][10] == 1:
              return 'K'
    elif result[0][11] == 1:
              return 'L'
    elif result[0][12] == 1:
              return 'M'
    elif result[0][13] == 1:
              return 'N'
    elif result[0][14] == 1:
              return 'O'
    elif result[0][15] == 1:
              return 'P'
    elif result[0][16] == 1:
              return 'Q'
    elif result[0][17] == 1:
              return 'R'
    elif result[0][18] == 1:
              return 'S'
    elif result[0][19] == 1:
              return 'T'
    elif result[0][20] == 1:
              return 'U'
    elif result[0][21] == 1:
              return 'V'
    elif result[0][22] == 1:
              return 'W'
    elif result[0][23] == 1:
              return 'X'
    elif result[0][24] == 1:
              return 'Y'
    elif result[0][25] == 1:
              return 'Z'


cap=cv2.VideoCapture(0)

cv2.namedWindow("TRACKBARS")

cv2.createTrackbar("L - H","TRACKBARS",0,179,nothing)
cv2.createTrackbar("L - S","TRACKBARS",0,255,nothing)
cv2.createTrackbar("L - V","TRACKBARS",0,255,nothing)
cv2.createTrackbar("U - H","TRACKBARS",179,179,nothing)
cv2.createTrackbar("U - S","TRACKBARS",255,255,nothing)
cv2.createTrackbar("U - V","TRACKBARS",255,255,nothing)

cv2.namedWindow("test")

img_counter=0

img_text=''

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    l_h = cv2.getTrackbarPos("L - H", "TRACKBARS")
    l_s = cv2.getTrackbarPos("L - S", "TRACKBARS")
    l_v = cv2.getTrackbarPos("L - V", "TRACKBARS")
    u_h = cv2.getTrackbarPos("U - H", "TRACKBARS")
    u_s = cv2.getTrackbarPos("U - S", "TRACKBARS")
    u_v = cv2.getTrackbarPos("U - V", "TRACKBARS")

    img=cv2.rectangle(frame,(425,100),(625,300),(0,255,0),2,lineType=8,shift=0)

    lower_blue=np.array([l_h,l_s,l_v])
    upper_blue=np.array([u_h,u_s,u_v])
    imcrop=img[102:298,427:623]
    hsv=cv2.cvtColor(imcrop,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower_blue,upper_blue)

    cv2.putText(frame,img_text,(100,400),cv2.FONT_HERSHEY_TRIPLEX,2,(255,255,0))
    cv2.imshow("test",frame)
    cv2.imshow("mask",mask)

    

    img_name='1.png'
    save_img=cv2.resize(mask,(image_x,image_y))
    cv2.imwrite(img_name,save_img)
    
    img_text=predictor()

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
