import cv2
import os
import argparse
from nt_model import model
from distance import *

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mouse_points = []


def get_mouse_coordinate(event, x, y, flags, param):

    global mouse_X, mouse_Y, mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_X, mouse_Y = x, y
        cv2.circle(image, (x, y), 0, (0, 0, 255), 12)
        if "mouse_points" not in globals():
            mouse_points = []
        mouse_points.append((x, y))
        print("Point marked")
        print(mouse_points)


# Command_line input setup
parser = argparse.ArgumentParser(description="SocialDistancing")
parser.add_argument(
    "--videopath", type=str, default="processing_video.mp4", help="video file path"
)
args = parser.parse_args()

processing_video = args.videopath

# Define a DNN_model
DNN = model()
# Get video handle
cap = cv2.VideoCapture(processing_video)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))

scale_width = 2 / 2
scale_height = 4 / 2

SOLID_BLACK_COLOR = (41, 41, 41)
# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_result = cv2.VideoWriter("Top_view.avi", fourcc, fps, (width, height))
bird_output = cv2.VideoWriter(
    "Video_bird.avi", fourcc, fps, (int(width * scale_width), int(height * scale_height))
)
# Initialize necessary variables
frame_number = 0
count_pedestrians_detected = 0
count_six_feet_violations = 0
all_pairs = 0
absolute_six_feet_violations = 0
pedestrian_per_second = 0
sh_index = 1
sc_index = 1

cv2.namedWindow("Video Processing")
cv2.setMouseCallback("Video Processing", get_mouse_coordinate)
number_mouse_points = 0
display_first_frame = True

# Process each frame, until end of video
while cap.isOpened():
    frame_number += 1
    ret, frame = cap.read()

    if not ret:
        print("Finished final Result:", str(int(count_six_feet_violations)), "violations")
        break

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    if frame_number == 1:
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        while True:
            image = frame
            cv2.imshow("Video Processing", image)
            cv2.waitKey(1)
            if len(mouse_points) == 7:
                cv2.destroyWindow("Video Processing")
                break
            display_first_frame = False
        four_points = mouse_points

        # Get perspective
        M, Minv = get_camera_perspective(frame, four_points[0:4])
        points = src = np.float32(np.array([four_points[4:]]))
        wrapped_point = cv2.perspectiveTransform(points, M)[0]
        threshhold_distance = np.sqrt(
            (wrapped_point[0][0] - wrapped_point[1][0]) ** 2
            + (wrapped_point[0][1] - wrapped_point[1][1]) ** 2
        )
        top_image = np.zeros(
            (int(frame_height * scale_height), int(frame_width * scale_width), 3), np.uint8
        )

        top_image[:] = SOLID_BLACK_COLOR
        pedestrian_detect = frame

    print("frame number", frame_number, ":", str(int(count_six_feet_violations)), "violations")

    # draw polygon of ROI
    points = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    )
    cv2.polylines(frame, [points], True, (255, 0, 0), thickness=4)

    # Detect person and bounding boxes using DNN
    pedestrian_boxes, number_pedestrians = DNN.detect_pedestrians(frame)

    if len(pedestrian_boxes) > 0:
        pedestrian_detect = plot_pedestrian_boxes_on_image(frame, pedestrian_boxes)
        wrapped_points, top_image = plot_points_on_bird_eye_view(
            frame, pedestrian_boxes, M, scale_width, scale_height
        )
        six_feet_violations, ten_feet_violations, pairs = plot_lines_between_nodes(
            wrapped_points, top_image, threshhold_distance
        )
        # plot_violation_rectangles(pedestrian_boxes, )
        count_pedestrians_detected += number_pedestrians
        all_pairs += pairs

        count_six_feet_violations += six_feet_violations / fps
        absolute_six_feet_violations += six_feet_violations
        pedestrian_per_second, sh_index = calculate_stay_at_home_index(
            count_pedestrians_detected, frame_number, fps
        )

    last_height =500
    text_display = "@Violations: " + str(int(count_six_feet_violations))
    pedestrian_detect, last_height = put_text(pedestrian_detect, text_display, text_offset_y=last_height)

    if all_pairs != 0:
        sc_index = 1 - absolute_six_feet_violations / all_pairs

    cv2.imshow("Street Camera", pedestrian_detect)
    cv2.waitKey(1)
    output_result.write(pedestrian_detect)
    bird_output.write(top_image)
