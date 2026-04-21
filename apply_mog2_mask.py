"""
apply_mog2_mask.py
------------------
Apply MOG2 background subtraction mask to a specific frame of a video and save the mask as an image.

Usage:
    python apply_mog2_mask.py <video_path> <output_mask_path> [<frame_number>]
    - <video_path>: Path to the input video file
    - <output_mask_path>: Path to save the output mask image (e.g., mask.png)
    - <frame_number>: (Optional) Frame number to process (default: 0, i.e., first frame)
"""

import sys
import cv2
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Usage: python apply_mog2_mask.py <video_path> <output_mask_path> [<frame_number>]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_mask_path = sys.argv[2]
    frame_number = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        sys.exit(1)

    # Move to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_number} from {video_path}")
        sys.exit(1)

    # Create MOG2 background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)

    # To get a meaningful mask, we need to apply the subtractor to several frames first
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(frame_number + 1):
        ret, f = cap.read()
        if not ret:
            print(f"Failed to read frame {i} from {video_path}")
            sys.exit(1)
        mask = bg_sub.apply(f)

    # Threshold the mask to get a binary image (as in detector.py)
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]

    # Save the mask image
    cv2.imwrite(output_mask_path, mask)
    print(f"Saved MOG2 mask for frame {frame_number} to {output_mask_path}")

    cap.release()

if __name__ == "__main__":
    main()
