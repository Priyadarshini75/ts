import cv2
from utils.distance_utils import calculate_distance

def analyze_crowd(frame, people_boxes, grid_size, frame_width, frame_height, crowd_threshold, distance_threshold):
    detected_crowds = []
    grid_width = frame_width // grid_size[0]
    grid_height = frame_height // grid_size[1]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            area_x1 = i * grid_width
            area_y1 = j * grid_height
            area_x2 = (i + 1) * grid_width
            area_y2 = (j + 1) * grid_height

            people_in_area = [person for person in people_boxes if area_x1 <= person[0][0] <= area_x2 and area_y1 <= person[0][1] <= area_y2]

            if len(people_in_area) > crowd_threshold:
                for a in range(len(people_in_area)):
                    for b in range(a + 1, len(people_in_area)):
                        if calculate_distance(people_in_area[a][0], people_in_area[b][0]) < distance_threshold:
                            detected_crowds.append((area_x1, area_y1, area_x2, area_y2))
                            cv2.rectangle(frame, (area_x1, area_y1), (area_x2, area_y2), (0, 0, 255), 2)
                            cv2.putText(frame, "Crowd Detected", (area_x1 + 10, area_y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            break
    return detected_crowds
