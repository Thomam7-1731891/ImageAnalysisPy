"""
Trey Michaels

This file is used to conduct data analysis involving image data to get
an idea of what image analysis can look like.
"""

import numpy as np
import imageio


def invert_colors(image):
    """
    Returns a new color image with all the colors inverted.

    Takes a color image and switches all higher values with lower values
    and vice versa for the image. Returned as a numpy array.
    """
    img = image
    invert = np.invert(img)
    return invert


def blur(image, patch):
    """
    Returns a new image that has been blurred.

    Takes a gray-scale image and a patch size to return a new numpy array
    that reflects an image that is blurred using the specified patch size.
    """
    img = image
    img_height, img_width = img.shape
    kernal = np.zeros((patch, patch))
    kernal = kernal + (1/(patch**2))
    result = np.zeros((img_height - patch + 1, img_width - patch + 1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            current = img[i:i+patch, j:j+patch]
            result[i, j] = np.sum(current * kernal)
    result = result.astype(np.uint8)
    return result


def template_match(large, small):
    """
    Returns a 2d numpy of floats that stores the similarity between the
    template at each point in the image.

    Takes am image as a 2d numpy array and a smaller image as a 2d numpy
    array to see if there is an instance of the smaller image found
    in the larger image.
    """
    image = large
    template = small
    image_height, image_width = image.shape
    template_height, template_width = template.shape
    result = np.zeros(
        (image_height - template_height + 1, image_width - template_width + 1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            current = image[i: i+template_height, j: j+template_width]
            current_avg = np.sum(current)/(template_height * template_width)
            current_demean = current - current_avg
            template_demean = template - current_avg
            result[i, j] = np.sum(current_demean * template_demean)
    return result


def main():
    image_puppy = imageio.imread("images/puppy.png")
    image_gray = imageio.imread("images/gray_puppy.png")
    test_array = np.array(
        [[1, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 1, 0]])
    test_template = np.array([[1, 0], [0, 1]])
    invert_colors(image_puppy)
    blur(image_gray, 40)
    template_match(test_array, test_template)


if __name__ == '__main__':
    main()
