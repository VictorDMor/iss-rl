import cv2
import numpy as np

def is_replay_screen(observation, template):
    # Convert to grayscale
    gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY) # Adjust the threshold value accordingly

    # Use cv2.matchTemplate to find the template in the thresholded image
    result = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Define a threshold for match quality (experiment to find a good value)
    threshold = 0.8

    if max_val > threshold:
        return True
    else:
        return False


if __name__ == '__main__':
    # Read the image file
    image_path = 'replay.png' # Change this to the path of your image
    image = cv2.imread(image_path)
    not_a_replay = cv2.imread('not_a_replay.png')
    template = cv2.imread('replay_template.png', cv2.IMREAD_GRAYSCALE)

    # Check if it's a replay screen
    is_replay = is_replay_screen(image, template)
    is_replay_2 = is_replay_screen(not_a_replay, template)
    print(is_replay, is_replay_2)
