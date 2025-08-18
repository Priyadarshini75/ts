import cv2
from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11s-seg.pt")  # or use another version as needed

# Set text prompt to detect specific classes
names = [
    "fire",
    "worker using mobile phone in restricted area",
    "car body on assembly line",
    "worker inspecting",
    "fire",
    "face mask",
    "safety jacket",
    "gloves",
    "car leaving property",
    "car entering property",
    "worker collapsing"
   ]
model.set_classes(names, model.get_text_pe(names))

# Open the video file
input_path = "input_path.mp4"
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_path = "output/output_annotated_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'/'avc1' based on platform
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOE prediction on the frame
    results = model.predict(source=frame, conf=0.25)

    # Convert the result to a frame with annotations
    annotated_frame = results[0].plot()  # .plot() returns the annotated image

    # Write the annotated frame to the output vide25o
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
print(f"Annotated video saved to {output_path}")