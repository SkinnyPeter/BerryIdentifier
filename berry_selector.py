import cv2
import numpy as np
import os
from datetime import datetime

# Adjustable parameters
CAMERA_PORT = 1
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS = 15
FLIP_IMAGE = False
BRIGHTNESS_COMPENSATION = -30  # New parameter to compensate for brightness
OUTPUT_DIR = 'captured_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BerryDetector:
    def __init__(self):
        # Video capture configuration
        self.cap = cv2.VideoCapture(CAMERA_PORT, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Color detection parameters (more refined HSV ranges)
        self.RED_LOWER1 = np.array([0, 90, 50])
        self.RED_UPPER1 = np.array([10, 255, 255])
        self.RED_LOWER2 = np.array([160, 90, 50])
        self.RED_UPPER2 = np.array([180, 255, 255])
        
        self.WHITE_LOWER = np.array([0, 0, 100])
        self.WHITE_UPPER = np.array([180, 50, 255])
        
        # Color rate tracking
        self.white_rate = 0.0
        self.red_rate = 0.0
        self.dominant_color = ''
    
    def calculate_color_rates(self, mask_red, mask_white, frame):
        # Calculate total frame area
        total_area = frame.shape[0] * frame.shape[1]
        
        # Calculate areas of detected colors
        red_area = np.sum(mask_red) / 255
        white_area = np.sum(mask_white) / 255
        
        # Calculate percentages
        self.red_rate = (red_area / total_area) * 100
        self.white_rate = (white_area / total_area) * 100
        
        # Determine dominant color
        rates = {
            'red': self.red_rate,
            'white': self.white_rate
        }
        self.dominant_color = max(rates, key=rates.get)
    
    def detect_berries(self, frame):
        # Adjust brightness if needed
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=BRIGHTNESS_COMPENSATION)
        
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create color masks with more robust detection
        mask_red1 = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1)
        mask_red2 = cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_white = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)
        
        # Calculate color rates
        self.calculate_color_rates(mask_red, mask_white, frame)
        
        # Create color-filtered frames
        red_filtered = cv2.bitwise_and(frame, frame, mask=mask_red)
        white_filtered = cv2.bitwise_and(frame, frame, mask=mask_white)
        
        # Create masks visualization
        masks = cv2.cvtColor(mask_red + mask_white, cv2.COLOR_GRAY2BGR)
        
        # Create filtered display
        filtered_display = np.hstack((red_filtered, white_filtered))
        
        # Prepare processed frame with detections
        processed = frame.copy()
        
        return processed, masks, filtered_display
    
    def add_text_overlay(self, display):
        # Prepare text for display
        text_lines = [
            f"Berry Detector",
            f"Dominant Color: {self.dominant_color.upper()}",
            f"Red Berries: {self.red_rate:.2f}%",
            f"White Berries: {self.white_rate:.2f}%"
        ]
        
        # Add text to the image
        for i, line in enumerate(text_lines):
            cv2.putText(display, line, (10, 30 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display
    
    def run(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect berries
            processed, masks, filtered_display = self.detect_berries(frame)

            # Display original frame and processed frame side by side
            display = np.hstack((frame, processed))
            
            # Add text overlay
            display = self.add_text_overlay(display)
            
            # Resize displays to have the same width
            common_width = 2 * FRAME_WIDTH
            display_resized = cv2.resize(display, (common_width, FRAME_HEIGHT))
            filtered_display_resized = cv2.resize(filtered_display, (common_width, FRAME_HEIGHT))

            # Combine all displays
            display_full = np.vstack((
                display_resized, 
                filtered_display_resized
            ))

            # Show the display
            cv2.imshow('Berry Detection', display_full)

            # Wait for a key press and handle 'q' or 'c'
            key = cv2.waitKey(1) & 0xFF

            # Break the loop if 'q' is pressed
            if key == ord('q'):
                print("Exiting..")
                break

            # Capture the image if 'c' is pressed
            if key == ord('c'):
                # We want to give the image a unique ID, lets use the date time
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Save the captured frame
                image_path = os.path.join(OUTPUT_DIR, f'captured_image_{timestamp}.jpg')
                cv2.imwrite(image_path, frame)
                print(f"Image saved at {image_path}")

            if key == ord('d'):
                # Log the berry detected - the dominant color
                if (self.dominant_color.upper() == 'RED'):
                    print("RED BERRY detected")
                
                if (self.dominant_color.upper() == 'WHITE'):
                    print("UNRIPE BERRY detected")
                    
        # Release everything when done
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        self.cap.release()

def main():
    detector = BerryDetector()
    detector.run()

if __name__ == "__main__":
    main()