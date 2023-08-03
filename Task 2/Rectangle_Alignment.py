# Importing the required libraries
import cv2
import numpy as np


# Function definition for aligning rectangles
def align_rectangles(image):
    # Input images converted into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold operation on grayscale image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Discovering contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Empty list to store information about rectangle images.
    aligned_images = []

    # Looping over each contour
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calculating the angle of rotation for aligning the rectangle.
        width, height = rect[1]
        if width > height:
            angle = rect[2]
        else:
            angle = rect[2] + 90

        # Finfding rotation matrix and applying transformation
        rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
        aligned_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        # Croping the aligned image to remove any borders or areas
        rect_points = cv2.transform(np.array([box]), rotation_matrix).squeeze().astype(int)
        x, y, w, h = cv2.boundingRect(rect_points)
        aligned_image = aligned_image[y:y + 2*h, x:x + 2*w]

        aligned_images.append(aligned_image)

    return aligned_images


# reading images
image = cv2.imread('task-2.png')
aligned_images = align_rectangles(image)

# displaying the aligned images
for i, aligned_image in enumerate(aligned_images):
    cv2.imshow(f'Aligned Image {i + 1}', aligned_image)

# closing displayed windows
cv2.waitKey(0)
cv2.destroyAllWindows()
