# AIM : 

# Write a program to perform basic image processing operations such as histogram equalization,
#  thresholding, edge detection, morphological operations , and apply basic data augmentation 
# techniques on a sample image.

# Algorithm:
# 1. Import the required libraries and load the input image.
# 2. Apply histogram equalization to enhance the image contrast.
# 3. Compute and display the histograms of the original and equalized images.
# 4. Apply Gaussian Blur and perform Canny edge detection.
# 5. Perform image augmentation using rotation, cropping, resizing, and horizontal flipping.
# 6. Convert the image into a binary image using thresholding.
# 7. Apply erosion and dilation using a structuring kernel.
# 8. Display the original and all processed images for comparison.

# Result :
# Basic image processing operations were successfully implemented using
# Python and OpenCV, producing the expected enhanced and transformed images.

from matplotlib import pyplot as plt
import cv2
import numpy as np

# Histogram Equalization

img = cv2.imread("Image_1.jfif", 0)

# Equalization
equ = cv2.equalizeHist(img)

# Histograms
hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([equ], [0], None, [256], [0, 256])

# Display images and histograms
plt.figure(figsize=(10,8))

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(222)
plt.imshow(equ, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(223)
plt.plot(hist1, color='blue')
plt.title('Original Histogram')

plt.subplot(224)
plt.plot(hist2, color='blue')
plt.title('Equalized Histogram')

plt.tight_layout()

plt.savefig("outputs/Histogram_Equalization.png", bbox_inches='tight')
plt.show()


# Edge Detection

img = cv2.imread("Image_1.jfif", 0)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blurred, 100, 200)

plt.figure(figsize=(10,8))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.axis('off')

plt.savefig("outputs/Edge_Detection_Result.png", bbox_inches='tight')
plt.show()


# Data Augmentation

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img = cv2.imread("Image_1.jfif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)

datagen = ImageDataGenerator(
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True
)

gen = datagen.flow(img, batch_size=1)

plt.figure(figsize=(10,10))

# Original image
plt.subplot(3,3,1)
plt.imshow(img[0])
plt.title("Original Image")
plt.axis('off')

# Augmented images
for i in range(8):
    plt.subplot(3,3,i+2)
    plt.imshow(next(gen)[0].astype('uint8'))
    plt.title(f"Augmented {i+1}")
    plt.axis('off')

plt.tight_layout()

plt.savefig("outputs/Augmented_Image.png", bbox_inches='tight')
plt.show()


# Morphological Operations

# Read image
img = cv2.imread("Image_1.jfif", 0)

# Convert to binary image
_, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Create kernel
kernel = np.ones((5, 5), np.uint8)

# Morphological operations
erosion = cv2.erode(binary_img, kernel, iterations=1)
dilation = cv2.dilate(binary_img, kernel, iterations=1)
opening = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

# Display results
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(binary_img, cmap='gray')
plt.title("Original Binary Image")
plt.axis('off')

plt.subplot(232)
plt.imshow(erosion, cmap='gray')
plt.title("Erosion")
plt.axis('off')

plt.subplot(233)
plt.imshow(dilation, cmap='gray')
plt.title("Dilation")
plt.axis('off')

plt.subplot(234)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis('off')

plt.subplot(235)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.axis('off')

plt.tight_layout()

plt.savefig("outputs/Morphological_Operation.png", bbox_inches='tight')
plt.show()