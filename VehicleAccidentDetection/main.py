import os
import argparse
from utils.model_loader import ModelLoader
from utils.accident_detector import AccidentDetector
from utils.video_processor import VideoProcessor

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Accident Detection System')
    parser.add_argument('--accident_model', type=str, default='models/best.pt', 
                        help='Path to accident detection model')
    parser.add_argument('--vehicle_model', type=str, default='models\yolov8n.pt', 
                        help='Path to vehicle detection model')
    parser.add_argument('--input_video', type=str, 
                        default='input_video/videoplayback (online-video-cutter.com).mp4',
                        help='Path to input video')
    parser.add_argument('--output_video', type=str, default='output_video/output.mp4',
                        help='Path to save output video')
    parser.add_argument('--screenshot_dir', type=str, default='screenshots',
                        help='Directory to save accident screenshots')
    parser.add_argument('--screenshot_cooldown', type=float, default=2.0,
                        help='Cooldown between screenshots in seconds')
    
    args = parser.parse_args()
    
    # Create screenshot directory
    os.makedirs(args.screenshot_dir, exist_ok=True)
    
    # Initialize components
    model_loader = ModelLoader(args.accident_model, args.vehicle_model)
    detector = AccidentDetector(model_loader)
    processor = VideoProcessor(
        input_path=args.input_video,
        output_path=args.output_video,
        detector=detector,
        screenshot_dir=args.screenshot_dir,
        screenshot_cooldown=args.screenshot_cooldown
    )
    
    # Process video
    processor.process()


if __name__ == "__main__":
    main()