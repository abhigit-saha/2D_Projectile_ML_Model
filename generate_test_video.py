import cv2
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH = "test_video3.mp4"  # Replace with your video file
OUTPUT_MASK_PATH = "mog2_mask_output.jpg"
OUTPUT_FRAME_PATH = "mog2_original_frame.jpg"

# The frame number to capture. Set this to a frame where the object 
# is actively moving across the screen (e.g., frame 30 or 50).
TARGET_FRAME = 30 

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        sys.exit(1)

    # Initialize MOG2 exactly as it is in your detector.py
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, 
        varThreshold=40, 
        detectShadows=False
    )

    frame_count = 0
    saved = False

    print(f"Processing video to frame {TARGET_FRAME} to build MOG2 history...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video before finding target frame.")
            break

        frame_count += 1

        # Apply MOG2 to build the background model
        # Learning rate is determined automatically by the history parameter
        fg_mask = bg_sub.apply(frame)

        # Apply the exact binary thresholding used in your pipeline
        # This converts grayscale certainty into a strict black/white binary mask
        _, binary_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Once we hit the target frame, save the outputs and break
        if frame_count == TARGET_FRAME:
            cv2.imwrite(OUTPUT_MASK_PATH, binary_mask)
            cv2.imwrite(OUTPUT_FRAME_PATH, frame) # Saving original for comparison
            print(f"Success! Saved frame {TARGET_FRAME}.")
            print(f"Mask saved to: {OUTPUT_MASK_PATH}")
            print(f"Original frame saved to: {OUTPUT_FRAME_PATH}")
            saved = True
            break

    cap.release()
    
    if not saved:
        print("Failed to save the target frame.")

if __name__ == "__main__":
    main()