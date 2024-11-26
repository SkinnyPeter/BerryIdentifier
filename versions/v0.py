import cv2
import numpy as np

# Adjustable parameters
CAMERA_PORT = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FLIP_IMAGE = False
BRIGHTNESS_COMPENSATION = 50  # New parameter to compensate for brightness

# Color detection parameters (in HSV)
RED_LOWER1 = np.array([0, 100, 100 - BRIGHTNESS_COMPENSATION])
RED_UPPER1 = np.array([10, 255, 255 - BRIGHTNESS_COMPENSATION])
RED_LOWER2 = np.array([160, 100, 100 - BRIGHTNESS_COMPENSATION])
RED_UPPER2 = np.array([180, 255, 255 - BRIGHTNESS_COMPENSATION])

BLACK_LOWER = np.array([0, 0, 0 + BRIGHTNESS_COMPENSATION])
BLACK_UPPER = np.array([180, 255, 50 + BRIGHTNESS_COMPENSATION])

WHITE_LOWER = np.array([0, 0, 200 - BRIGHTNESS_COMPENSATION])
WHITE_UPPER = np.array([180, 30, 255 - BRIGHTNESS_COMPENSATION])

MIN_CIRCULARITY = 0.7
MIN_AREA = 100
MAX_AREA = 5000

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 2
BOX_THICKNESS = 2

class BerryDetector:
    def __init__(self):
        # Video capture configuration
        self.cap = cv2.VideoCapture(CAMERA_PORT)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
    
    def detect_berries(self, frame):
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust color thresholds based on brightness
        mask_red1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask_red2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_black = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)
        mask_white = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
        
        processed = frame.copy()
        self.process_mask(mask_red, frame, processed, "Red berry", (0, 0, 255))
        self.process_mask(mask_black, frame, processed, "Black berry", (0, 0, 0))
        self.process_mask(mask_white, frame, processed, "White berry", (255, 255, 255))
        
        if FLIP_IMAGE:
            processed = cv2.flip(processed, 1)
        
        # Create an image with color masks, resized to match main display width
        masks = np.hstack((mask_red, mask_black, mask_white))
        masks = cv2.cvtColor(masks, cv2.COLOR_GRAY2RGB)
        masks = cv2.resize(masks, (FRAME_WIDTH * 2, FRAME_HEIGHT // 2))
        
        return processed, masks
    
    def process_mask(self, mask, original, processed, label, color):
        # Find and process contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_AREA < area < MAX_AREA:
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > MIN_CIRCULARITY:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(original, (x, y), (x + w, y + h), color, BOX_THICKNESS)
                    cv2.rectangle(processed, (x, y), (x + w, y + h), color, BOX_THICKNESS)
                    cv2.putText(original, label, (x, y - 10), FONT, FONT_SCALE, color, FONT_THICKNESS)
                    cv2.putText(processed, label, (x, y - 10), FONT, FONT_SCALE, color, FONT_THICKNESS)
    
    def run(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect berries
            processed, masks = self.detect_berries(frame)

            # Display original frame and processed frame side by side
            display = np.hstack((frame, processed))
            
            # Display color masks below
            display_full = np.vstack((display, masks))

            # Show the display
            cv2.imshow('Berry Detection', display_full)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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