import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import mss
from PIL import Image


def getimage():
    with mss.mss() as sct:
        monitors=sct.monitors
        monitor=monitors[2]
        screenshot=sct.grab(monitor)
        screenshot = Image.fromarray(np.array(screenshot)).convert("RGB")
        screenshot = screenshot.resize((480,270))
        return screenshot



def get_road_mask(img):
    """Extract red-colored road mask from image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red has two ranges in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    return red_mask

def is_on_road(road_mask):
    """Check if the car (center bottom of image) is on the road."""
    h, w = road_mask.shape
    roi_y_start = int(h * 0)
    roi_y_end = int(h * 0.4)
    roi_x_center = w // 2
    roi_width = w // 10
    roi = road_mask[roi_y_start:roi_y_end, roi_x_center - roi_width//2 : roi_x_center + roi_width//2]
    road_ratio = np.sum(roi == 255) / roi.size

    return road_ratio > 0.5

def recover_towards_road(road_mask):
    """Decide whether to turn left or right based on road location."""
    h, w = road_mask.shape
    roi_y_start = int(h * 0)
    roi_y_end = int(h * 0.6)
    roi_x_center = w // 2
    # roi_width = w // 6

    left_roi = road_mask[roi_y_start:roi_y_end, :roi_x_center]
    right_roi = road_mask[roi_y_start:roi_y_end, roi_x_center:]

    left_score = np.sum(left_roi == 255)
    right_score = np.sum(right_roi == 255)

    return 1 if left_score > right_score else 2


if __name__ == '__main__':
	import time
	while True:
		# if not keyboard.is_pressed("q"):
		# 	continue
		# if keyboard.is_pressed("esc"):
		# 	break
		img = getimage()
		img = np.array(img)
		road_mask = get_road_mask(img)
		if is_on_road(road_mask):
			print("Yes")
		else:
			print(recover_towards_road(road_mask))
		cv2.imshow("vis", img)
		cv2.imshow("mask", road_mask)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		time.sleep(0.5)