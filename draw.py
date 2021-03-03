import cv2
import numpy as np

h,w = 512,512
img = np.zeros((h,w,1), np.uint8)
# cv2.imshow("blank image", blank_image)

drawing = False # true if mouse is pressed
# mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def mouse_callback_fn(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            # if mode == True:
            #     cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            # else:
            #     cv2.circle(img,(x,y),5,(0,0,255),-1)

            cv2.line(img,(ix,iy),(x,y),255,2)
            ix,iy = x,y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # if mode == True:
        #     cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
        # else:
        #     cv2.circle(img,(x,y),5,(0,0,255),-1)


cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse_callback_fn)

while(1):
    # print(filled)
    # cv2.imshow('filled',filled)
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mask = np.zeros((h+2,w+2),np.uint8)
        num,filled,mask,rect = cv2.floodFill(img,mask,(int(h/2),int(w/2)),(255))
        # print(len(filled))
        img = filled
    elif k == 27:
        mask = np.zeros((h+2,w+2),np.uint8)
        num,filled,mask,rect = cv2.floodFill(img,mask,(int(h/2),int(w/2)),(255))
        img = filled
        img = cv2.bitwise_not(img)
        cv2.imwrite("shape.png",img)

        resized = cv2.resize(img,(32,32), interpolation=cv2.INTER_CUBIC)
        input_image = np.ones((128,64,1), np.uint8)*255

        pos_y,pos_x = 32,50

        h1,w1 = resized.shape[0], resized.shape[1]

        # print(int(pos_y-h1/2),int(pos_x-w1/2))
        roi = input_image[int(pos_x-w1/2):int(pos_x-w1/2)+w1,int(pos_y-h1/2):int(pos_y-h1/2)+h1,0]
        # print(roi.shape)

        input_image[int(pos_x-w1/2):int(pos_x-w1/2)+w1,int(pos_y-h1/2):int(pos_y-h1/2)+h1,0] = resized
        cv2.imwrite("input.png",input_image)

        input_image = input_image/255.0
        np.save("input",input_image)

        break


cv2.destroyAllWindows()
