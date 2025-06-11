"""
First example using this script for presenting frame-differencing.
"""

import os
import cv2 as cv
import glob

from algorithms.background_subtraction import process_frames


def load_images_from_directory(directory_path):
    """Load all PNG images in the specified directory into OpenCV format."""
    image_files = glob.glob(os.path.join(directory_path, "*.png"))
    images = []
    for file_path in image_files:
        # Load image in color (BGR format)
        image = cv.imread(file_path, cv.IMREAD_COLOR)
        print(f"Info: image with shape {image.shape} read")
        if image is not None:
            images.append((file_path, image))
        else:
            print(f"Warning: Failed to load image '{file_path}'")
    return images

def main():
    CUTS = {"1": "1.1", "2": "1.2", "3": "1.3", "4": "2.1", "5": "2.2"}
    DIRECTORY = "/Users/enoks/Programming/VROOM/training-set/moving-camera/25mm/2.1/"

    if not os.path.isdir(DIRECTORY):
        print(f"Error: '{DIRECTORY}' is not a valid directory.")
        return

    images = sorted(load_images_from_directory(DIRECTORY))
    print(f"Loaded {len(images)} image(s).")

    # Example of processing: display dimensions
    # for path, img in images:
    #     print(f"{path}: shape = {img.shape}")  # (height, width, channels)
        
    #     key = cv.waitKey(1000)
    #     if  key == ord('q'):
    #         break
    #     elif key == ord('s'):
    #         r = cv.selectROI('Info: please select a region', img)
    #         ok = tracker.init(img, r)
    #         print(f"Info: region {r} selected")

    #     cv.imshow(f'frame@{path}', img)

    index = 12

    frame_0 = images[index][1]
    frame_1 = images[index + 1][1]

    frame_0_gray = cv.cvtColor(frame_0, cv.COLOR_BGR2GRAY)
    frame_1_gray = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)
    
    cv.imshow("", frame_0)
    cv.waitKey(10000)

    cv.imshow("", frame_1)
    cv.waitKey(10000)

    result_0 = cv.absdiff(frame_0_gray, frame_1_gray)

    cv.imshow("Preview", result_0)
    cv.waitKey(10000)
        
    cv.imwrite("/Users/enoks/Downloads/example-2_frame-0.jpg", frame_0)
    cv.imwrite("/Users/enoks/Downloads/example-2_frame-1.jpg", frame_1)
    cv.imwrite("/Users/enoks/Downloads/example-2_result-0.jpg", result_0)

if __name__ == "__main__":
    main()
