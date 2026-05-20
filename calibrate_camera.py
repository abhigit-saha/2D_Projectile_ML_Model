import cv2
import numpy as np
import glob
import os
import argparse

def calibrate_camera(image_dir, checkerboard_size=(9, 6), square_size=0.025):
    """
    Calibrate the camera using a set of checkerboard images.
    
    Args:
        image_dir: Directory containing calibration images (.jpg, .png, etc.)
        checkerboard_size: Tuple (inner_corners_x, inner_corners_y)
        square_size: Size of a single square in meters (default 25mm)
    """
    # Termination criteria for subpixel corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(image_dir, '*.*'))
    if not images:
        print(f"No images found in {image_dir}")
        return None

    print(f"Found {len(images)} images for calibration.")
    
    img_shape = None
    success_count = 0

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            success_count += 1
            
            # Optional: Draw and display the corners
            # cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
            
    # cv2.destroyAllWindows()

    if success_count == 0:
        print("Could not find checkerboard in any images. Check your grid size!")
        return None

    print(f"Successfully found corners in {success_count} images. Calibrating...")

    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\n--- Calibration Results ---")
    print(f"RMS Error: {ret:.4f} pixels")
    print("\nCamera Matrix (K):")
    print(mtx)
    print("\nDistortion Coefficients:")
    print(dist)
    
    print("\nAdd this to your config.py:")
    print("CAMERA_K = np.array([")
    print(f"    [{mtx[0,0]:.2f}, 0.0, {mtx[0,2]:.2f}],")
    print(f"    [0.0, {mtx[1,1]:.2f}, {mtx[1,2]:.2f}],")
    print(f"    [0.0, 0.0, 1.0]")
    print("])")
    
    return mtx, dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate camera using checkerboard images.")
    parser.add_argument('--dir', type=str, default='calibration_images', help='Directory with images')
    parser.add_argument('--grid', type=str, default='9,6', help='Inner corners (x,y)')
    parser.add_argument('--size', type=float, default=0.025, help='Square size in meters')
    
    args = parser.parse_args()
    gx, gy = map(int, args.grid.split(','))
    
    if not os.path.exists(args.dir):
        print(f"Directory {args.dir} does not exist. Please create it and add images.")
    else:
        calibrate_camera(args.dir, (gx, gy), args.size)
