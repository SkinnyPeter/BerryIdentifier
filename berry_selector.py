import cv2
import numpy as np
import os
from datetime import datetime

# Adjustable parameters
CAMERA_PORT = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FLIP_IMAGE = False
BRIGHTNESS_COMPENSATION = 50  # New parameter to compensate for brightness
OUTPUT_DIR = 'captured_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BerryDetector:
    def __init__(self):
        # Video capture configuration
        self.cap = cv2.VideoCapture(CAMERA_PORT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Color detection parameters (more refined HSV ranges)
        self.RED_LOWER1 = np.array([0, 100, 100])
        self.RED_UPPER1 = np.array([10, 255, 255])
        self.RED_LOWER2 = np.array([160, 100, 100])
        self.RED_UPPER2 = np.array([180, 255, 255])
        
        self.BLACK_LOWER = np.array([0, 0, 0])
        self.BLACK_UPPER = np.array([180, 100, 50])
        
        self.WHITE_LOWER = np.array([0, 0, 200])
        self.WHITE_UPPER = np.array([180, 50, 255])
        
        # Parameters for detection
        self.MIN_CIRCULARITY = 0.7
        self.MIN_AREA = 50
        self.MAX_AREA = 5000
        
        # Color rate tracking
        self.white_rate = 0.0
        self.black_rate = 0.0
        self.red_rate = 0.0
        self.dominant_color = ''
    
    def calculate_color_rates(self, mask_red, mask_black, mask_white, frame):
        # Calculate total frame area
        total_area = frame.shape[0] * frame.shape[1]
        
        # Calculate areas of detected colors
        red_area = np.sum(mask_red) / 255
        black_area = np.sum(mask_black) / 255
        white_area = np.sum(mask_white) / 255
        
        # Calculate percentages
        self.red_rate = (red_area / total_area) * 100
        self.black_rate = (black_area / total_area) * 100
        self.white_rate = (white_area / total_area) * 100
        
        # Determine dominant color
        rates = {
            'red': self.red_rate,
            'black': self.black_rate,
            'white': self.white_rate
        }
        self.dominant_color = max(rates, key=rates.get)
    
    def detect_berries(self, frame):
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create color masks with more robust detection
        mask_red1 = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1)
        mask_red2 = cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_black = cv2.inRange(hsv, self.BLACK_LOWER, self.BLACK_UPPER)
        mask_white = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)
        
        # Calculate color rates
        self.calculate_color_rates(mask_red, mask_black, mask_white, frame)
        
        # Create color-filtered frames
        red_filtered = cv2.bitwise_and(frame, frame, mask=mask_red)
        black_filtered = cv2.bitwise_and(frame, frame, mask=mask_black)
        white_filtered = cv2.bitwise_and(frame, frame, mask=mask_white)
        
        # Prepare processed frame with detections
        processed = frame.copy()
        self.process_mask(mask_red, frame, processed, "Red berry", (0, 0, 255))
        self.process_mask(mask_black, frame, processed, "Black berry", (0, 0, 0))
        self.process_mask(mask_white, frame, processed, "White berry", (255, 255, 255))
        
        if FLIP_IMAGE:
            processed = cv2.flip(processed, 1)
            red_filtered = cv2.flip(red_filtered, 1)
            black_filtered = cv2.flip(black_filtered, 1)
            white_filtered = cv2.flip(white_filtered, 1)
        
        # Prepare mask visualization
        masks = np.hstack((mask_red, mask_black, mask_white))
        masks = cv2.cvtColor(masks, cv2.COLOR_GRAY2RGB)
        masks = cv2.resize(masks, (FRAME_WIDTH * 2, FRAME_HEIGHT // 2))
        
        # Prepare color-filtered frames for display
        filtered_display = np.hstack((red_filtered, black_filtered, white_filtered))
        filtered_display = cv2.resize(filtered_display, (FRAME_WIDTH * 3, FRAME_HEIGHT))
        
        return processed, masks, filtered_display
    
    def process_mask(self, mask, original, processed, label, color):
        # Find and process contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.MIN_AREA < area < self.MAX_AREA:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > self.MIN_CIRCULARITY:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(original, (x, y), (x + w, y + h), color, 2)
                    cv2.rectangle(processed, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(processed, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def add_text_overlay(self, display):
        # Prepare text for display
        text_lines = [
            f"Berry Detector",
            f"Dominant Color: {self.dominant_color.upper()}",
            f"Red Berries: {self.red_rate:.2f}%",
            f"Black Berries: {self.black_rate:.2f}%", 
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
            masks_resized = cv2.resize(masks, (common_width, FRAME_HEIGHT // 2))
            filtered_display_resized = cv2.resize(filtered_display, (common_width, FRAME_HEIGHT))

            # Combine all displays
            display_full = np.vstack((
                display_resized, 
                masks_resized, 
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