import cv2
from PIL import Image, ImageOps
import pytesseract
import numpy as np
import util


class ImageProcessor:

    def __init__(self, image_path, output_path):

        # Initiation
        self.output_path = output_path
        self.originalImage = cv2.imread(image_path)
        self.image = cv2.copyMakeBorder(self.originalImage, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
        self.image_height = len(self.image)
        self.image_width = len(self.image[0])

        blue, green, red = cv2.split(self.image)

        # Setting the threshold for edge detection for RGB parts
        blue_edges = cv2.Canny(blue, 128, 255)
        green_edges = cv2.Canny(green, 128, 255)
        red_edges = cv2.Canny(red, 128, 255)

        # Join back into one image
        self.edges = blue_edges | green_edges | red_edges

        # Find the contours and their hierarchy
        self.contours, hierarchy = cv2.findContours(self.edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.hierarchy = hierarchy[0]

        # Contours that survive the process of elimination
        self.contour_containers_of_interest = []

        # Here we store the registration gathered by single character recognition
        self.registration_cumulative = ""

        self.registration = ""

    def process(self):

        # First round of elimination
        # Looping through all the contours
        for index, contour in enumerate(self.contours):
            # Get the bounding rectangle of the given contour
            x, y, width, height = cv2.boundingRect(contour)

            # Natural selection
            if self.evaluate(contour) and self.include_box(index, contour):
                self.contour_containers_of_interest.append([contour, [x, y, width, height]])

        # Output image is a white image of the same size as the original
        output_image = self.edges.copy()
        output_image.fill(255)

        avg = 0

        # Get the average height of all remaining contours
        for index, (contour, box) in enumerate(self.contour_containers_of_interest):
            x_, y_, width, height = box
            avg += height

        avg /= len(self.contour_containers_of_interest)

        # Remove all contours shorter than the average height
        self.contour_containers_of_interest = list(filter(lambda contour_container: contour_container[1][3] > avg,
                                                          self.contour_containers_of_interest))

        heights = []

        for index, (contour, box) in enumerate(self.contour_containers_of_interest):
            x, y, width, height = box
            heights.append(height)

        median = np.median(heights)

        # From the remaining contours, eliminate all who's height deviates too much from the median
        self.contour_containers_of_interest = list(filter(lambda k: k[1][3] - (avg / 2) < median < k[1][3] + (avg / 2),
                                                          self.contour_containers_of_interest))

        # Sort contours from left to right for single character recognition
        self.contour_containers_of_interest = sorted(self.contour_containers_of_interest, key=lambda k: (k[1][0]))

        counter = 0

        for index, (contour, box) in enumerate(self.contour_containers_of_interest):

            foreground_intensity = 0.0
            # Calculate the average pixel intensity of the contour
            for position in contour:
                foreground_intensity += util.perceived_luminance(self.image, position[0][0], position[0][1])

            foreground_intensity /= len(contour)

            x_, y_, width, height = box

            # Calculate the average background pixel intensity around the contour
            background_intensity = util.calculate_box_background_pixel_intensity(self.image, box)

            # Which color is the foreground and which the background
            # To be applied to contour pixels and background pixels
            if foreground_intensity >= background_intensity:
                background_color = 0
                foreground_color = 255
            else:
                background_color = 255
                foreground_color = 0

            # Invert the pixel color if necessary
            for x in range(x_, x_ + width):
                for y in range(y_, y_ + height):
                    if y >= self.image_height or x >= self.image_width:
                        continue
                    if util.perceived_luminance(self.image, x, y) > foreground_intensity:
                        output_image[y][x] = background_color
                    else:
                        output_image[y][x] = foreground_color

            # Get the cropped image according to the bounding box
            crop_image = output_image[y_: y_ + height, x_: x_ + width]
            crop_image = ImageOps.expand(Image.fromarray(crop_image), border=5, fill='white')
            cv2.imshow("cropped", np.asarray(crop_image))
            # Uncomment the following line to see every contour bounding box after processing
            # cv2.waitKey(0)

            # Most serbian registrations have 7 characters, 2 letters, 3 digits and 2 letters
            # If we have 7 boudning boxes as a result of processing, treat them as letters/digits in recognition
            if len(self.contour_containers_of_interest) == 7:
                if counter < 2 or counter > 4:
                    addition = pytesseract.image_to_string(crop_image, config='-c tessedit_char_whitelist=ASDFGHJKLQWERTYUIOPMNBVCXZ -psm 10')
                else:
                    addition = pytesseract.image_to_string(crop_image, config='-c tessedit_char_whitelist=1234567890 -psm 10')

                self.registration_cumulative += addition
                counter += 1
            else:
                self.registration_cumulative += pytesseract.image_to_string(crop_image, config='-c tessedit_char_whitelist=ASDFGHJKLQWERTYUIOPMNBVCXZ1234567890 -psm 10')

        # Write the resulting image in a file
        cv2.imwrite(self.output_path, output_image)

        # Run recognition on the resulting image
        self.registration = pytesseract.image_to_string(Image.fromarray(output_image), config='-c tessedit_char_whitelist=ASDFGHJKLQWERTYUIOPMNBVCXZ0123456789')
        self.registration = self.registration.replace(" ", "")
        print (self.registration_cumulative)
        print (self.registration)

        print ("----- FINAL RESULT -----")
        # Should the final result be the recognition of the whole output image, or the accumulated characters
        if len(self.registration_cumulative) == len(self.registration) or "\n" in self.registration:
            print (self.registration_cumulative)
        else:
            print (self.registration)

    # Invoking utility functions for contour elimination
    def evaluate(self, contour):
        return util.is_contour_closed(contour) and util.keep_box(self.image, contour)

    # Eliminating contours based on the number of children of their parents
    def include_box(self, index, contour):

        # If a contour has more than 5 child contours, discard, since it's likely not a character
        if self.count_children(index, contour) > 5:
            return False

        # If a contour is a child, and the parent has less than 5 children, discard (license plates usually have at
        # least 5 characters)
        if self.is_child(index) and self.count_children(self.get_parent(index), contour) <= 5:
            return False

        return True

    def count_children(self, index, contour):
        # Check to see weather the contour has any children in the hierarchy
        if self.hierarchy[index][2] < 0:
            return 0
        else:
            if self.evaluate(self.contours[self.hierarchy[index][2]]):
                count = 1
            else:
                count = 0

            count += self.count_contours_on_the_same_hierarchy_level(self.hierarchy[index][2], contour)
            return count

    def is_child(self, index):
        return self.get_parent(index) > 0

    def get_parent(self, index):
        parent = self.hierarchy[index][3]
        while not self.evaluate(self.contours[parent]) and parent > 0:
            parent = self.hierarchy[parent][3]

        return parent

    def count_contours_on_the_same_hierarchy_level(self, index, contour):

        count = self.count_children(index, contour)

        # Count all contours on the "right" of the initial contour
        next_contour_index = self.hierarchy[index][0]
        while next_contour_index > 0:
            if self.evaluate(self.contours[next_contour_index]):
                count += 1
            count += self.count_children(next_contour_index, contour)
            next_contour_index = self.hierarchy[next_contour_index][0]

        # Count all contours on the "left" of the initial contour
        previous_contour_index = self.hierarchy[index][1]
        while previous_contour_index > 0:
            if self.evaluate(self.contours[previous_contour_index]):
                count += 1
            count += self.count_children(previous_contour_index, contour)
            previous_contour_index = self.hierarchy[previous_contour_index][1]
        return count
