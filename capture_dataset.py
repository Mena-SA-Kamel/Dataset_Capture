import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import cv2
from PIL import Image
from dataset_planner import get_num_objects_per_image, initialize_objects

def to_text(objects_list):
    text = "Objects needed: "
    for object in objects_drawn:
        text = text + object + ', '
    return text

## Setting up work directories

dataset_name = 'Refined Dataset test'
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

    folders = ['train', 'validate', 'test']
    subfolders = ['rgb', 'depth']
    for folder in folders:
        folder_path = os.path.join(dataset_name, folder)
        os.mkdir(folder_path)
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            os.mkdir(subfolder_path)

log_file = open(os.path.join(dataset_name,'log_file.txt'), 'a')

dataset_type = 'train'

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 60)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)
colorizer = rs.colorizer()
frame_count = 0

# Initial set of objects
objects = initialize_objects()
objects_drawn = np.random.choice(objects, get_num_objects_per_image())
image_text = to_text(objects_drawn)


for i in list(range(20)):
    frames = pipeline.wait_for_frames()
# Streaming loop
try:
    while True:
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        # depth_color_frame = colorizer.colorize(aligned_depth_frame)
        hole_filling = rs.hole_filling_filter()
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)

        if not aligned_depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # depth_image = depth_image * (depth_image < 2000)

        # import code; code.interact(local=dict(globals(), **locals()))
        if float(np.max( depth_image )) == 0:
            continue
        depth_scaled = ((depth_image / float(np.max( depth_image ))) * 255).astype('uint8')

        rgbd_image = np.zeros([color_image.shape[0], color_image.shape[1], 4])
        rgbd_image[:, :, 0:3] = color_image
        rgbd_image[:, :, 3] = depth_scaled

        depth_3_channel = cv2.cvtColor(depth_scaled,cv2.COLOR_GRAY2RGB)

        # bgr -> rgb
        color_image_new = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        images = np.hstack((color_image_new, depth_3_channel))
        key = cv2.waitKey(1)

        if key == 32:
            black_image = np.zeros([color_image.shape[0], color_image.shape[1], 3])
            black_view = np.hstack((black_image, black_image))
            images = black_view
            # cv2.imshow('Align Example', black_view)
            current_time = datetime.now()
            image_name = current_time.strftime("%Y-%m-%d-%H-%M-%S.png")
            # import code; code.interact(local=dict(globals(), **locals()))
            color_image_path = os.path.join(dataset_name, dataset_type, 'rgb', image_name)
            depth_image_path = os.path.join(dataset_name, dataset_type, 'depth', image_name)

            color_image = Image.fromarray(color_image)
            depth_scaled = Image.fromarray(depth_scaled)
            color_image.save(color_image_path)
            depth_scaled.save(depth_image_path)

            entry_name = image_name + "   " + str(objects_drawn) + '\n'
            log_file.write(entry_name)


            objects_drawn = np.random.choice(objects, get_num_objects_per_image())
            image_text = to_text(objects_drawn)

        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.putText(images, image_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
        cv2.imshow('Align Example', images)



        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            log_file.close()
            break
finally:
    pipeline.stop()
