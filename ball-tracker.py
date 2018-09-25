# deque is a list-like data structure for fast appends and pops
from collections import deque

import numpy as np
import cv2
import imutils
import time

#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the
    # list of tracked points
    greenLower = (35, 0, 51)
    greenUpper = (99, 255, 240)
    pts = deque(maxlen=64)

    vs = cv2.VideoCapture(0)

    # allow the camera or video file to warm up
    # time.sleep(2.0)

    # keep looping
    while True:
        # grab the current frame
        (grabbed, frame) = vs.read()

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        center = None

        colour = 255

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
        pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            colour = (colour+1)%255

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)


        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break


# free up memory
vs.release()
cv2.destroyAllWindows()
