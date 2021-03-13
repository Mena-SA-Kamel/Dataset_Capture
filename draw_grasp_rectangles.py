import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
import time
import copy
import skimage.io
import random

last_time_r = None
last_time_l = None
last_time_m = None
undo = False
next_box = False


def generate_random_colors():
    r = random.random()
    g = random.random()
    b = random.random()
    return (r,g,b)

def visualize_boxes(image, grasping_points):
    past_id = 0
    color = generate_random_colors()
    img_to_display = image.copy()
    for grasping_box in grasping_points:
        instance_id = grasping_box[-1]
        if past_id != instance_id:
            color = generate_random_colors()
            past_id = instance_id
        box = cv2.boxPoints(tuple(grasping_box[:-1]))
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, color, 2)
    cv2.imshow('grasping_boxes', image)
    cv2.waitKey(0)



events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

dataset_name = 'Refined Dataset'
dataset_type = 'train'
image_type = 'rgb'
training_set_path = os.path.join(dataset_name, dataset_type, image_type)
label_folder_path = os.path.join(dataset_name, dataset_type, 'label')
grasping_points_folder_path = os.path.join(dataset_name, dataset_type, 'grasping_points')
instance_id = 0

if not os.path.exists(grasping_points_folder_path):
    os.mkdir(grasping_points_folder_path)

image_names = os.listdir(training_set_path)
for image_name in image_names:
    if image_name == '.DS_Store':
        continue
    if os.path.exists(os.path.join(grasping_points_folder_path, image_name.replace('.png', '.txt'))):
        # if not image_name.replace('.png', '.txt') == '2020-05-08-21-49-47.txt':
        #     continue
        continue

    print ('Currently labelling: ', image_name)

    label_path = os.path.join(label_folder_path, image_name)
    label_image = skimage.io.imread(label_path)
    label_image = np.array(label_image)

    if len(np.unique(label_image)) == 0:
        continue

    img = mpimg.imread(os.path.join(training_set_path,image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    instance_mask = (label_image == instance_id)
    instance_mask = instance_mask.astype('float32')
    instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_GRAY2RGB)
    masked_image = cv2.addWeighted(img, 0.8, instance_mask, 0.2, 0)
    previous_image = copy.copy(masked_image)

    cv2.namedWindow('image',cv2.WND_PROP_FULLSCREEN)

    kill = False
    points = []
    point_storage = []
    grasp_rectangles = []
    draw = False
    undo = False
    instance_id = 0

    # mouse callback function
    def draw_rectangle(event,x,y,flags,param):
        global kill
        global points
        global point_storage
        global draw
        global previous_image
        global last_time_l
        global last_time_r
        global last_time_m
        global undo
        global img
        global masked_image
        global instance_id
        global grasp_rectangles
        global next_box
        kill = False

        if event == cv2.EVENT_LBUTTONDOWN:
            if last_time_l is not None and time.time() - last_time_l < 1:
                draw = True
                last_time_l = None
            else:
                last_time_l = time.time()


        # if event == cv2.EVENT_RBUTTONDOWN:
        #     if last_time_r is not None and time.time() - last_time_r < 1:
        if next_box:
            points = [points]
            if len(points) > 0 :
                for point in points:
                    point.append(point[0])
                    point = np.array(point)
                    try:
                        rotated_rectangle = cv2.minAreaRect(point)
                        rectangle_vertices = cv2.boxPoints(rotated_rectangle)
                        rectangle_vertices = np.int0(rectangle_vertices)
                        cv2.drawContours(masked_image, [rectangle_vertices], 0, (0, 0, 255), 2)
                        rotated_rectangle = np.array(rotated_rectangle)
                        rotated_rectangle = np.append(rotated_rectangle, instance_id)
                        grasp_rectangles =np.append(grasp_rectangles,rotated_rectangle)
                        grasp_rectangles = grasp_rectangles.reshape(-1, 4)
                        cv2.imshow('image', masked_image)
                        cv2.waitKey(1)
                    except:
                        continue


                previous_image = copy.copy(masked_image)
                points = []
                point_storage = []
                draw = False
                last_time_r = None
                next_box = False
            # else:
            #     last_time_r = time.time()

        if draw:
            cv2.circle(masked_image, (x, y), 1, (255,0,0), -1)
            points.append([x, y])

        if undo:
            img = copy.copy(previous_image)
            points = []
            undo = False
            draw = False
            grasp_rectangles = np.delete(grasp_rectangles, -1, axis=0)


    while(1):
        cv2.setMouseCallback('image', draw_rectangle)
        cv2.imshow('image',masked_image)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('b'):
            print 'Clear Image. This image needs to be relabelled: ', image_name
            img = mpimg.imread(os.path.join(training_set_path, image_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('image', cv2.WND_PROP_FULLSCREEN)
            kill = False
            points = []
            point_storage = []
            grasp_rectangles = []
            draw = False
            undo = False
            instance_id = 0
        if k == ord('s'):
            print 'Skipped: ', image_name
            break
        if k == ord('u'):
            undo = True
            print 'undo: ', image_name

        if k == ord('n') or (not instance_id in np.unique(label_image)):
            # Save image and go to next one
            instance_id = 0
            file_path = os.path.join(grasping_points_folder_path, image_name.replace('.png', '.txt'))

            with open(file_path, "ab") as f:
                np.savetxt(f, grasp_rectangles, fmt="%s")
            visualize_boxes(img, grasp_rectangles)
            f.close()
            cv2.destroyAllWindows()
            break
        if k == ord('u'):
            undo = True
            print 'undo: ', image_name
        if k == ord('g'):
            instance_id += 1
            print('Label the next instance')
            instance_mask = (label_image == instance_id)
            instance_mask = instance_mask.astype('float32')
            instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_GRAY2RGB)
            masked_image = cv2.addWeighted(img, 0.5, instance_mask, 0.5, 0)

        if k == 32:
            next_box = True


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

