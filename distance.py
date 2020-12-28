import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform


def plot_lines_between_nodes(wrapped_points, top_image, threshhold_distance):
    points = np.array(wrapped_points)
    dist_condense = pdist(points, 'minkowski', p=1.8)
    distance = squareform(dist_condense)

    # 10 feet calculation
    cal_distance = np.where(distance < threshhold_distance * 6 / 10)
    close_point = []
    color_10 = (0, 0, 255)
    lineThick = 4
    tenfeet_violations = len(np.where(dist_condense < 10 / 6 * threshhold_distance)[0])
    for i in range(int(np.ceil(len(cal_distance[0]) / 2))):
        if cal_distance[0][i] != cal_distance[1][i]:
            point1 = cal_distance[0][i]
            point2 = cal_distance[1][i]

            close_point.append([point1, point2])

            cv2.line(
                top_image,
                (points[point1][0], points[point1][1]),
                (points[point2][0], points[point2][1]),
                color_10,
                lineThick,
            )



    # 6 feet calculation
    cal_distance = np.where(distance < threshhold_distance)
    sixfeet_violations = len(np.where(dist_condense < threshhold_distance)[0])
    total_pairs = len(dist_condense)
    danger_point = []
    color_6 = (0, 0, 255)
    for i in range(int(np.ceil(len(cal_distance[0]) / 2))):
        if cal_distance[0][i] != cal_distance[1][i]:
            point1 = cal_distance[0][i]
            point2 = cal_distance[1][i]

            danger_point.append([point1, point2])
            cv2.line(
                top_image,
                (points[point1][0], points[point1][1]),
                (points[point2][0], points[point2][1]),
                color_6,
                lineThick,
            )

    # display top-view
    cv2.imshow("top-view", top_image)
    cv2.waitKey(1)

    return sixfeet_violations, tenfeet_violations, total_pairs


def plot_points_on_bird_eye_view(frame, pedestrian_boxes, Mark, scale_width, scale_height):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    radius_of_node = 7
    node_color = (41, 41, 41)
    thickness_of_node = 20
    solid_black_color = (255, 255, 255)

    blank_img = np.zeros(
        (int(frame_height * scale_height), int(frame_width * scale_width), 3), np.uint8
    )
    blank_img[:] = solid_black_color
    wrapped_pts = []
    for i in range(len(pedestrian_boxes)):

        mid_point_x = int(
            (pedestrian_boxes[i][1] * frame_width + pedestrian_boxes[i][3] * frame_width) / 2
        )
        mid_point_y = int(
            (pedestrian_boxes[i][0] * frame_height + pedestrian_boxes[i][2] * frame_height) / 2
        )

        points = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        wrapped_pt = cv2.perspectiveTransform(points, Mark)[0][0]
        warped_point_scaled = [int(wrapped_pt[0] * scale_width), int(wrapped_pt[1] * scale_height)]

        wrapped_pts.append(warped_point_scaled)
        top_image = cv2.circle(
            blank_img,
            (warped_point_scaled[0], warped_point_scaled[1]),
            radius_of_node,
            node_color,
            thickness_of_node,
        )

    return wrapped_pts, top_image


def get_camera_perspective(img, src_points):
    Image_h = img.shape[0]
    Image_w = img.shape[1]
    source = np.float32(np.array(src_points))
    distance = np.float32([[0, Image_h], [Image_w, Image_h], [0, 0], [Image_w, 0]])

    Mark = cv2.getPerspectiveTransform(source, distance)
    Mark_inv = cv2.getPerspectiveTransform(distance, source)

    return Mark, Mark_inv


def put_text(frame, text, text_offset_y=25):
    font_scale_per = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    rectangle_background = (255, 255, 255)
    (text_w, text_h) = cv2.getTextSize(
        text, font, fontScale=font_scale_per, thickness=1
    )[0]
    # set the text_display start position
    text_offset_x = frame.shape[1] - 550
    # make the coords of the box with a small padding of two pixels
    box_coordinate = (
        (text_offset_x, text_offset_y + 5),
        (text_offset_x + text_w + 2, text_offset_y - text_h - 2),
    )
    frame = cv2.rectangle(
        frame, box_coordinate[0], box_coordinate[1], rectangle_background, cv2.FILLED
    )
    frame = cv2.putText(
        frame,
        text,
        (text_offset_x, text_offset_y),
        font,
        fontScale=font_scale_per,
        color=(41, 41, 41),
        thickness=2,
    )

    return frame, 2 * text_h + text_offset_y

def calculate_stay_at_home_index(total_pedestrians_detected, frame_num, fps):
    normal_people = 10
    pedestrians = np.round(total_pedestrians_detected / frame_num, 1)
    sh_idx = 1 - pedestrians / normal_people
    return pedestrians, sh_idx


def plot_pedestrian_boxes_on_image(frame, pedestrian_boxes):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    thick = 2

    node_color = (0, 0, 255)

    for i in range(len(pedestrian_boxes)):
        point1 = (
            int(pedestrian_boxes[i][1] * frame_width),
            int(pedestrian_boxes[i][0] * frame_height),
        )
        point2 = (
            int(pedestrian_boxes[i][3] * frame_width),
            int(pedestrian_boxes[i][2] * frame_height),
        )

        frame_boxes = cv2.rectangle(frame, point1, point2, node_color, thick)


    return frame_boxes
