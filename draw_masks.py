import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import copy

last_time_r = None
last_time_l = None
last_time_m = None
undo = False


events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

dataset_name = 'Refined Dataset'
dataset_type = 'train'
image_type = 'rgb'
training_set_path = os.path.join(dataset_name, dataset_type, image_type)
label_folder_path = os.path.join(dataset_name, dataset_type, 'label')

if not os.path.exists(label_folder_path):
    os.mkdir(label_folder_path)

image_names = os.listdir(training_set_path)
for image_name in image_names:
    if image_name == '.DS_Store':
        continue
    if os.path.exists(os.path.join(label_folder_path, image_name)):
        continue
    print 'Currently labelling: ', image_name
    img = mpimg.imread(os.path.join(training_set_path,image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    previous_image = copy.copy(img)
    cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)

    kill = False
    multi_region = False
    objects = []
    points = []
    point_storage = []
    mask = np.zeros(img.shape[0:2])
    num_objects = 0
    draw = False
    undo = False

    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global kill
        global points
        global point_storage
        global multi_region
        global draw
        global previous_image
        global num_objects
        global mask
        global last_time_l
        global last_time_r
        global last_time_m
        global undo
        global img
        kill = False

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_time_l is not None and time.time() - last_time_l < 1:
                draw = True
                last_time_l = None
            else:
                last_time_l = time.time()

        if event == cv2.EVENT_RBUTTONDOWN:
            if last_time_r is not None and time.time() - last_time_r < 1:
                num_objects = num_objects + 1
                if multi_region:
                    point_storage.append(points)
                    points = point_storage
                else:
                    points = [points]
                for point in points:
                    point.append(point[0])
                    point = np.array(point)
                    # cv2.polylines(img, [point], True, (255, 255, 255))
                    cv2.fillPoly(img, [point], (255, 255, 255))
                    mask = cv2.fillPoly(mask, [point], num_objects)
                objects.append(points)
                previous_image = copy.copy(img)
                points = []
                point_storage = []
                draw = False
                last_time_r = None
                # kill = True
            else:
                last_time_r = time.time()

        if draw:
            cv2.circle(img, (x, y), 1, (255,0,0), -1)
            points.append([x, y])
            if event == cv2.EVENT_MBUTTONDOWN:
                if last_time_m is not None and time.time() - last_time_m < 1:
                    points.append(points[0])
                    point_storage.append(points)
                    points = []
                    multi_region = True
                    last_time_m = None
                    draw = False
                else:
                    last_time_m = time.time()

        if undo:
            img = copy.copy(previous_image)
            points = []
            point_storage = []
            undo = False
            draw = False


    while(1):
        cv2.setMouseCallback('image', draw_circle)
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            print 'Clear Image. This image needs to be relabelled: ', image_name
            img = mpimg.imread(os.path.join(training_set_path, image_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
            kill = False
            multi_region = False
            objects = []
            points = []
            point_storage = []
            mask = np.zeros(img.shape[0:2])
            num_objects = 0
            draw = False
            undo = False
        if k == ord('s'):
            print 'Skipped: ', image_name
            break
        if k == ord('u'):
            undo = True
            print 'undo: ', image_name
        if k == ord('m'):
            mask = mask.astype('uint8')
            label_image = Image.fromarray(mask)
            label_image.save(os.path.join(label_folder_path, image_name))
            plt.imshow(mask)
            plt.show()
        if k == ord('n'):
            # Save image and go to next one
            mask = mask.astype('uint8')
            label_image = Image.fromarray(mask)
            label_image.save(os.path.join(label_folder_path, image_name))
            plt.imshow(mask)
            plt.show()
            cv2.destroyAllWindows()
            break
        if k == ord('u'):
            undo = True
            print 'undo: ', image_name

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 15)
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2

        cv2.putText(img, image_name,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

