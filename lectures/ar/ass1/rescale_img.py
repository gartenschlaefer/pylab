import numpy as np
import cv2 as cv
import glob

from skimage.transform import rescale, resize, downscale_local_mean


#images = glob.glob('./ignore/checker-board/origin/*.jpg')
images = glob.glob('./ignore/EasyMarker/origin/*.jpg')

#out_path = './ignore/checker-board/'

out_path = './ignore/EasyMarker/'

out_fname = 'easy'

for i, fname in enumerate(images):

    img = cv.imread(fname)
    #print(img)

    x_px = 600
    y_px = round(600 / img.shape[1] * img.shape[0])

    rescale = resize(img, (y_px, x_px), anti_aliasing=True)
    #print(img)

    print("image: ", fname)
    print("size: ", img.shape)
    print("rsize: ", rescale.shape)

    cv.imwrite(out_path + out_fname + str(i) + '.jpg', rescale * 255)

