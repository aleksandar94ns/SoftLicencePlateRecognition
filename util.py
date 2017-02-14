import cv2
import numpy as np


# Perceived pixel intensity
def perceived_luminance(image, x, y):
    if x >= len(image[0]) or y >= len(image):
        return 0
    pixel = image[y][x]
    return 0.2126 * pixel[2] + 0.7125 * pixel[1] + 0.0722 * pixel[0]


# If the contour is not closed, it's unlikely to be the bound of a character, so we discard it
def is_contour_closed(contour):
    first = contour[0][0]
    last = contour[len(contour) - 1][0]
    return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1


# Eliminating contours based on the bounding box size in respect to
# the expected proportions and size of a license plate character
def keep_box(image, contour):
    x, y, width, height = cv2.boundingRect(contour)

    # Converting to float so Python stops crying
    width *= 1.0
    height *= 1.0

    # If the bounding box is not wide enough or too wide to be a letter, discard it
    if width / height < 0.15 or width / height > 1:
        return False

    # If the bounding box is close to a square, discard it
    if 0.7 < width / height < 1.3:
        return False

    # If the bounding box is too huge or too small, discard it
    if ((width * height) > ((len(image[0]) * len(image)) / 5)) or ((width * height) < 25):
        return False

    return True


def calculate_box_background_pixel_intensity(image, box):
    x, y, width, height = box

    bg_int = \
        [
            # top left corner
            perceived_luminance(image, x - 1, y + height + 1),
            perceived_luminance(image, x - 1, y + height),
            perceived_luminance(image, x, y + height + 1),

            # top right corner
            perceived_luminance(image, x + width + 1, y + height + 1),
            perceived_luminance(image, x + width, y + height + 1),
            perceived_luminance(image, x + width + 1, y + height),

            # bottom left corner
            perceived_luminance(image, x - 1, y - 1),
            perceived_luminance(image, x - 1, y),
            perceived_luminance(image, x, y - 1),

            # bottom right corner
            perceived_luminance(image, x + width + 1, y - 1),
            perceived_luminance(image, x + width, y - 1),
            perceived_luminance(image, x + width + 1, y)
        ]

    return np.median(bg_int)
