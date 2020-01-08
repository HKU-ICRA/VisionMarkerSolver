import cv2
import numpy as np

from detect import detectMarker


# Define markers world position here
squareLength = 16.0
markers_pos = {
    # 'marker_name': [ [] ]
    'one': np.array([[-squareLength / 2 , squareLength /  2, 0],
            [squareLength / 2, squareLength / 2, 0],
            [squareLength / 2, -squareLength / 2, 0],
            [-squareLength / 2, -squareLength / 2, 0]])
}

# Camera matrix
fx, fy = 3.5, 3.5
cx, cy = 0, 0
camMat = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0 , 1]
])

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    matches = detectMarker(frame)
    for m in matches:
        if m[0] in markers_pos:
            retval, rvec, tvec = cv2.solvePnP(markers_pos[m[0]], m[1], camMat, distCoeffs=None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            print("Marker: " + m[0])
            print("R-vec: ", rvec)
            print("T-vec: ", tvec)
            print("\n")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
