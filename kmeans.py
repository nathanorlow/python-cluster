import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np

NUMBER_OF_COLORS = 4
FILENAME = "flamingo"
EXTENSION = ".png"
#read in the image -- a each entry in the array is a pixel (3 dimensional array)
input_image=mpimg.imread(FILENAME + EXTENSION)
[height, width, channel_count] = input_image.shape
pixel_count = height*width

#convert the array of pixels to a list of pixels (so it can be passed to kmeans)
image_column = input_image.reshape([pixel_count, channel_count])
#then run kmeans on the pixels
kmeans_column_result = KMeans(n_clusters=NUMBER_OF_COLORS, random_state=0, n_init="auto").fit(image_column)

#The clusters form the "palette" for image
label_to_color = kmeans_column_result.cluster_centers_

#Each label indicates the palette color to paint that pixel
all_labels = kmeans_column_result.labels_.reshape(pixel_count)

#for each pixel, get the label and lookup that color in the palette (label_to_color)
output_image_column = np.zeros([pixel_count,channel_count])
for index in range(0,pixel_count) :
    label = all_labels[index]
    color = label_to_color[label]
    output_image_column[index] = color

#convert it back to a two dimensional image
output_image = output_image_column.reshape(height, width, channel_count)

#display the original image and output image
image_plot = plt.imshow(output_image)
plt.axis('off')
plt.savefig(FILENAME + "-output-" + str(NUMBER_OF_COLORS) + EXTENSION, bbox_inches='tight')