import numpy as np
import matplotlib.pyplot as plt

# results_file = open("colab_result_8b_July_16, 2020.txt", "r")
results_file = open("combined_architechture_results_colab_attempt#21-September28, 2020.txt", "r")
epochs = results_file.read().splitlines()
epochs = np.array(epochs)

losses = []
rpn_class_losses = []
rpn_bbox_losses = []
mrcnn_class_losses = []
mrcnn_bbox_losses = []
mrcnn_mask_losses = []
grasp_losses = []
val_losses = []
val_rpn_class_losses = []
val_rpn_bbox_losses = []
val_mrcnn_class_losses = []
val_mrcnn_mask_losses = []
val_grasp_losses = []
val_mrcnn_bbox_losses = []

for i, epoch in enumerate(epochs):
    try:
        metrics = epoch.split(' - ')
        loss = float(metrics[2].split(':')[1])
        rpn_class_loss = float(metrics[3].split(':')[1])
    except:
        print(epoch)
        continue
    # rpn_class_loss = float(metrics[3].split(':')[1])
    try:
        rpn_bbox_loss = float(metrics[4].split(':')[1])
        mrcnn_class_loss = float(metrics[5].split(':')[1])
        mrcnn_bbox_loss = float(metrics[6].split(':')[1])
        mrcnn_mask_loss = float(metrics[7].split(':')[1])
        grasp_loss = float(metrics[8].split(':')[1])
        val_loss = float(metrics[9].split(':')[1])
        val_rpn_class_loss = float(metrics[10].split(':')[1])
        val_rpn_bbox_loss = float(metrics[11].split(':')[1])
        val_mrcnn_class_loss = float(metrics[12].split(':')[1])
        val_mrcnn_bbox_loss = float(metrics[13].split(':')[1])
        val_mrcnn_mask_loss = float(metrics[14].split(':')[1])
        val_grasp_loss = float(metrics[15].split(':')[1])

        losses.append(loss)
        rpn_class_losses.append(rpn_class_loss)
        rpn_bbox_losses.append(rpn_bbox_loss)
        mrcnn_class_losses.append(mrcnn_class_loss)
        mrcnn_bbox_losses.append(mrcnn_bbox_loss)
        mrcnn_mask_losses.append(mrcnn_mask_loss)
        grasp_losses.append(grasp_loss)
        val_losses.append(val_loss)
        val_rpn_class_losses.append(val_rpn_class_loss)
        val_rpn_class_losses.append(val_rpn_class_loss)
        val_mrcnn_class_losses.append(val_mrcnn_class_loss)
        val_mrcnn_mask_losses.append(val_mrcnn_mask_loss)
        val_grasp_losses.append(val_grasp_loss)
        val_rpn_bbox_losses.append(val_rpn_bbox_loss)
        val_mrcnn_bbox_losses.append(val_mrcnn_bbox_loss)
    except:
        continue


val_grasp_losses = np.array(val_grasp_losses)
epoch_numbers = np.arange(val_grasp_losses.shape[0])
val_rpn_bbox_loss = np.array(val_rpn_bbox_loss)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(epoch_numbers, grasp_losses, label='Grasp loss')
# ax1.plot(epoch_numbers, losses, color='r', label='Total loss')
# ax1.plot(epoch_numbers, mrcnn_mask_losses, color='g', label='Mask loss')
# ax1.axvspan(500, 1000, color='g', alpha=0.5)
# ax1.text(100,0.090,'LR = 0.002')
# ax1.text(201,0.090,'LR = LR x 10')
# ax1.text(250,0.090,'LR = 0.0002')
ax1.set_title('Training Loss')
ax1.legend()
# ax1.set_xlim([0, 100])
# ax1.set_ylim([0.6, 1.2])

ax2.plot(epoch_numbers, val_grasp_losses, label='Grasp loss')
# ax2.plot(epoch_numbers, val_losses, color='r', label='Total loss')
# ax2.plot(epoch_numbers, val_mrcnn_mask_losses, color='g', label='Mask loss')
# ax2.axvspan(500, 1000, color='g', alpha=0.5)
# ax2.text(100,0.090,'LR = 0.002')
# ax2.text(201,0.090,'LR = LR x 10')
# ax2.text(250,0.090,'LR = 0.0002')
ax2.set_title('Validation Loss')
ax2.legend()
# ax2.set_xlim([0, 100])
# ax2.set_ylim([0.6, 1.2])

# ax3.plot(epoch_numbers, val_rpn_bbox_losses, label='Grasp loss')
# # ax2.plot(epoch_numbers, val_losses, color='r', label='Total loss')
# # ax2.plot(epoch_numbers, val_mrcnn_mask_losses, color='g', label='Mask loss')
# # ax2.axvspan(500, 1000, color='g', alpha=0.5)
# # ax2.text(100,0.090,'LR = 0.002')
# # ax2.text(201,0.090,'LR = LR x 10')
# # ax2.text(250,0.090,'LR = 0.0002')
# ax3.set_title('Val RPN Loss')
# ax3.legend()
# ax3.set_xlim([0, 150])
# ax3.set_ylim([0, 10])
#
# ax4.plot(epoch_numbers, val_mrcnn_bbox_losses, label='Grasp loss')
# # ax2.plot(epoch_numbers, val_losses, color='r', label='Total loss')
# # ax2.plot(epoch_numbers, val_mrcnn_mask_losses, color='g', label='Mask loss')
# # ax2.axvspan(500, 1000, color='g', alpha=0.5)
# # ax2.text(100,0.090,'LR = 0.002')
# # ax2.text(201,0.090,'LR = LR x 10')
# # ax2.text(250,0.090,'LR = 0.0002')
# ax4.set_title('Val RPN Loss')
# ax4.legend()
# ax4.set_xlim([0, 150])
# ax4.set_ylim([0, 10])

ax1.set(xlabel='Epochs', ylabel='Loss')
ax2.set(xlabel='Epochs', ylabel='Loss')

fig.tight_layout(pad=1.0)
ax1.grid()
ax2.grid()

plt.show()
results_file.close()
