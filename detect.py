import cv2
import numpy as np

from markerDict import MarkerDict


markerDict = MarkerDict()


def orderRects(rects):
    pts_sum = rects.sum(axis=2)
    pts_diff = np.diff(rects, axis=2)
    orderedRects = np.zeros((rects.shape[0], 4, 2), dtype=np.float32)
    for idx in range(rects.shape[0]):
        orderedRects[idx, 0] = rects[idx, np.argmin(pts_sum[idx])]      # top left
        orderedRects[idx, 2] = rects[idx, np.argmax(pts_sum[idx])]      # bot right
        orderedRects[idx, 1] = rects[idx, np.argmin(pts_diff[idx])]     # top right
        orderedRects[idx, 3] = rects[idx, np.argmax(pts_diff[idx])]     # bot left
    return orderedRects


def perspectiveTrans(img, orderedRects):
    N = orderedRects.shape[0]
    tl = orderedRects[:, 0]
    tr = orderedRects[:, 1]
    br = orderedRects[:, 2]
    bl = orderedRects[:, 3]
    w1 = np.sqrt((br[:, 0] - bl[:, 0])**2 + (br[:, 1] - bl[:, 1])**2)
    w2 = np.sqrt((tr[:, 0] - tl[:, 0])**2 + (tr[:, 1] - tl[:, 1])**2)
    ws = np.max([w1, w2], axis=0).astype(dtype=np.int32)
    h1 = np.sqrt((tr[:, 0] - br[:, 0])**2 + (tr[:, 1] - br[:, 1])**2)
    h2 = np.sqrt((tl[:, 0] - bl[:, 0])**2 + (tl[:, 1] - bl[:, 1])**2)
    hs = np.max([h1, h2], axis=0).astype(dtype=np.int32)
    dsts = [np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32) for w, h in zip(ws, hs)]
    trans = [cv2.getPerspectiveTransform(rect, dst) for rect, dst in zip(orderedRects, dsts)]
    warps = [cv2.warpPerspective(img, tran, (w, h)) for tran, w, h in zip(trans, ws, hs)]
    return warps


def squareImage(ratio, img):
    H = img.shape[0]
    W = img.shape[1]
    M = H // ratio  # Height
    N = W // ratio  # Width
    HDiff = H - M * 7
    WDiff = W - N * 7
    if HDiff % 2 == 0:
        HL = HR = HDiff // 2
    else:
        HL = HDiff // 2 + 1
        HR = HDiff // 2
    if WDiff % 2 == 0:
        WL = WR = WDiff // 2
    else:
        WL = WDiff // 2 + 1
        WR = WDiff // 2
    img = img[HL:H-HR, WL:W-WR]
    return img


def tile_debug(warp):
    ratio = 7
    warp = squareImage(ratio, warp)
    H = warp.shape[0]
    W = warp.shape[1]
    M = H // ratio
    N = W // ratio
    for y in range(0, H, M):
        for x in range(0, W, N):
            yh = y + M
            xw = x + N
            cv2.rectangle(warp, (x, y), (xw, yh), (0, 255, 0))
    return warp


def tile(warp):
    ratio = 7
    warp = squareImage(ratio, warp)
    H = warp.shape[0]
    W = warp.shape[1]
    M = H // ratio
    N = W // ratio
    grid = [warp[y:y+M, x:x+N] for y in range(0, H, M) for x in range(0, W, N)]
    return grid


def detectMarker(img):
    matches = []
    img_og = img
    ''' Threshold image '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.inRange(img, (25, 25, 25), (30, 255, 255))
    cv2.imshow("Image", img)
    ''' Get contours '''
    contours, heirarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursSorted = sorted(contours, key=lambda x: -cv2.contourArea(x))
    ''' Get rectangles '''
    epsilon = 0.15 * cv2.arcLength(contoursSorted[0], True)    # Five percent of the largest contour
    rects = [cv2.approxPolyDP(contour, epsilon, True) for contour in contoursSorted]
    #cv2.drawContours(img_og, rects[0], -1, (0, 255, 0), 3)
    #cv2.imshow("Img og", img_og)
    rects = [rect for rect in rects if len(rect) == 4]
    rects = np.asarray(rects[0:3])  # Take first three potential rectangles
    if len(rects.shape) < 4:
        return matches
    rects = np.squeeze(rects, axis=2)
    ''' Prune rectangles by removing those within existing rectangles '''
    #TO DO
    ''' Perspective transform  '''
    orderedRects = orderRects(rects)
    warps = perspectiveTrans(img, orderedRects)
    ''' Split image into 7x7 tiles '''
    grids = [tile(warp) for warp in warps if warp.shape[0] >= 7 and warp.shape[1] >= 7]   # Prune by area
    if warps[0].shape[0] >= 7 and warps[0].shape[1] >= 7:
        deimg = tile_debug(warps[0])
        cv2.imshow("DeImage", deimg)
    ''' Better padding / cropping to preserve symmetry '''
    #TO DO
    ''' Determine whether cell is filled '''
    markers = [[(cv2.countNonZero(g) / 49) >= 0.8 for g in grid] for grid in grids]
    for idx, marker in enumerate(markers):
        for k, v in markerDict.markers.items():
            if marker == v:
                matches.append((k, orderedRects[idx]))
    return matches


#img = cv2.imread("./markertest.jpg")
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    matches = detectMarker(frame)
    if len(matches) > 0:
        print(matches)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
