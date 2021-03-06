import cv2
import os
import time

import tensorflow as tf

from face_detector import FaceDetector
import argparse

parser = argparse.ArgumentParser(description='Face Detector')
parser.add_argument('--input', type=str, help='Enter path to an input image')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
detector = FaceDetector(['./FD_quantized_freeze.pb'])



def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "pts":
            #     continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)
    return fileList


def facedetect(args):
    count = 0
    data_dir = args.input
    pics = []
    GetFileList(data_dir,pics)

    pics = [x for x in pics if 'jpg' in x or 'png' in x]
    #pics.sort()

    for pic in pics:

        img=cv2.imread(pic)

        img_show = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        star=time.time()
        boxes=detector(img,0.5)
        #print('one iamge cost %f s'%(time.time()-star))
        #print(boxes.shape)
        #print(boxes)
        ################toxml or json


        print(boxes.shape[0])
        if boxes.shape[0]==0:
            print(pic)
        for box_index in range(boxes.shape[0]):

            bbox = boxes[box_index]

            cv2.rectangle(img_show, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)


        cv2.namedWindow('res',0)
        cv2.imshow('res',img_show)
        cv2.waitKey(0)
    print(count)


if __name__=='__main__':
    args = parser.parse_args()

    facedetect(args)
