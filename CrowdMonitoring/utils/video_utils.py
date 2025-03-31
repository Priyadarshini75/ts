import cv2

def initialize_video_writer(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    return cap, out, frame_width, frame_height

def release_resources(cap, out):
    cap.release()
    out.release()
    cv2.destroyAllWindows()
