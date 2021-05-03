import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

location = "../extra_material/result_images/batch_3"

# img = cv2.imread(os.path.join(location,"2.png"))
# cv2.namedWindow("image",cv2.WINDOW_NORMAL)
# cv2.imshow("image", img)
# cv2.waitKey(0)

def crop_image(image_name):

    img = cv2.imread(os.path.join(location,image_name))
    w,h = 149,297
    
    # shape = img[184:184+h,135:135+w]
    # shape_mini = img[184+18:184+18+64,135:135+w]

    # u_true = img[184:184+h,357:357+w]
    # v_true = img[184:184+h,579:579+w]
    # p_true = img[184:184+h,801:801+w]

    # u_pred = img[568:568+h,357:357+w]
    # v_pred = img[568:568+h,579:579+w]
    # p_pred = img[568:568+h,801:801+w]

    # u_error = img[952:952+h,357:357+w]
    # v_error = img[952:952+h,579:579+w]
    # p_error = img[952:952+h,801:801+w]

    img2 = img[184:568+h][135:801+w]
    cv2.imwrite(os.path.join(new_folder,image_name), img2)

    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image", img2)
    cv2.waitKey(0)


    print("[INFO] saved file "+image_name)


new_folder = os.path.join(location,"zoomed")
# try:
#     os.makedirs(new_folder)
# except:
#     print("Directory exists")

# all_images = os.listdir(location)
# print(all_images)
# for image in all_images:
#     crop_image(image)

crop_image("2.png")