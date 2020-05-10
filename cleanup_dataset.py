import os
from shutil import copyfile
from shutil import move



dataset_name_new = 'New Graspable Objects Dataset'
if not os.path.exists(dataset_name_new):
    os.mkdir(dataset_name_new)
    folders = ['train', 'validate', 'test']
    subfolders = ['rgb', 'depth', 'label']
    for folder in folders:
        folder_path = os.path.join(dataset_name_new, folder)
        os.mkdir(folder_path)
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            os.mkdir(subfolder_path)

dataset_name = 'Graspable Objects Dataset'
dataset_name = dataset_name_new
folders = ['train', 'validate', 'test']
subfolders = ['rgb', 'depth', 'label']

training_images_path = os.path.join(dataset_name, 'train')
validating_images_path = os.path.join(dataset_name, 'validate')
testing_images_path = os.path.join(dataset_name, 'test')

training_images = os.listdir(os.path.join(training_images_path, 'rgb'))
validating_images = os.listdir(os.path.join(validating_images_path, 'rgb'))
testing_images = os.listdir(os.path.join(testing_images_path, 'rgb'))

training_images_path_new = os.path.join(dataset_name_new, 'train')
validating_images_path_new = os.path.join(dataset_name_new, 'validate')
testing_images_path_new = os.path.join(dataset_name_new, 'test')
#
# for image in training_images:
#     for subfolder in subfolders:
#         src = os.path.join(training_images_path, subfolder, image)
#         if (not os.path.exists(src)) and subfolder == 'label':
#             continue
#         dst = os.path.join(training_images_path_new, subfolder, image)
#         copyfile(src, dst)
#
# for image in validating_images:
#     for subfolder in subfolders:
#         src = os.path.join(validating_images_path, subfolder, image)
#         if (not os.path.exists(src)) and subfolder == 'label':
#             continue
#         dst = os.path.join(validating_images_path_new, subfolder, image)
#         copyfile(src, dst)
#
# for image in testing_images:
#     for subfolder in subfolders:
#         src = os.path.join(testing_images_path, subfolder, image)
#         if (not os.path.exists(src)) and subfolder == 'label':
#             continue
#         dst = os.path.join(testing_images_path_new, subfolder, image)
#         copyfile(src, dst)

#
for image in testing_images:
    src = os.path.join(validating_images_path, 'label', image)
    if (not os.path.exists(src)):
        continue
    dst = os.path.join(testing_images_path, 'label', image)
    move(src, dst)
