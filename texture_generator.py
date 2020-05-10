import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

def rescale_depth(depth_image):
    max_val = np.max(depth_image)
    if len(np.unique(depth_image)) == 1:
        return depth_image
    min_val = np.unique(depth_image)[1]
    pixel_range = max_val - min_val
    scale = 1 / pixel_range
    depth_image = depth_image * scale
    object_depth = depth_image / np.max(depth_image)
    return object_depth

def get_textures(image, image_mask):
    texture_source_rgb = plt.imread('image_000003.png')
    texture_source_mask = plt.imread('image_000003_mask.png')
    texture_source_depth = plt.imread('image_000003_depth.png')

    mask_ids = np.unique(texture_source_mask)
    counter = 0
    textures = []
    for i in mask_ids:
        if counter == 0:
            counter = counter + 1
            continue
        region_pixels = (texture_source_mask == i)
        region_pixels = np.array(region_pixels)

        mask = np.zeros_like(texture_source_rgb)
        for i in range(3):
            mask[:, :, i] = region_pixels.copy()

        object_depth = mask * texture_source_depth
        object_depth = rescale_depth(object_depth)

        object_color = mask * texture_source_rgb
        textures.append(object_color)
        counter = counter + 1


    image_mask_ids = np.unique(image_mask)
    for i in image_mask_ids:
        colorization_mask = (image_mask == i)
        to_colorize = image * colorization_mask
        to_colorize_rescaled = rescale_depth(to_colorize)

        # import code;
        # code.interact(local=dict(globals(), **locals()))


        W = image.shape[1]
        H = image.shape[0]
        texture = io.imread('texture_test.png')
        texture = resize(texture, (H, W),
                               anti_aliasing=True)

        to_colorize_rescaled = to_colorize_rescaled * 255
        to_colorize_rescaled = to_colorize_rescaled.astype('uint8')

        to_colorize_reshaped = cv2.cvtColor(to_colorize_rescaled,cv2.COLOR_GRAY2RGB)

        lowpass = ndimage.gaussian_filter(to_colorize_reshaped, 6)
        gauss_highpass = to_colorize_reshaped - lowpass

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Horizontally stacked subplots')
        ax1.imshow(to_colorize_reshaped)
        ax2.imshow(gauss_highpass)
        ax3.imshow((texture * (6*to_colorize_reshaped/255.0)))
        plt.show()
        #
        # import code;
        # code.interact(local=dict(globals(), **locals()))


to_texturize = color.rgb2gray(io.imread('image_000039_depth.png'))
masks =  color.rgb2gray(io.imread('image_000039_mask.png'))
get_textures(to_texturize, masks)