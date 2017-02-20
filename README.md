# Advanced-Lane-Lines

Advanced Lane Finding Project

The goals / steps of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("birds-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/5aba544c-13b9-440f-b35c-f67aef7e2946

https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md



Camera calibration:
Distortion causes changes in objects appearance (size, shape and relative distance). The first step in analyzing camera images, is to undo this distortion so that we can get correct and useful information out of them. 
To correct distortion we will take pictures of known shapes (usually it will be a chessboard - because it has regular high contrast patterns) from the camera we are using and then correct distortion errors.
Here, we will use provided camera images (folder: ) of a chessboard and use cv2 functions to correct distortions
Take multiple images and compare with chessboard on a flat surface
Map distorted points to undistorted points

Read in chessboard images with corners 9x6
Map coordinates of corners of 2D image points to real 3D object points
Detect corners using findChessboardCorners (which retruns corners found in a gray scale image)
Append corners retruned in previous step to image points array
cv2.calibrateCamera to calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

Use distortion coefficients and camera matrix returned from previous step along with cv2.undistort to return an undistorted image

Summary : 
Lane curvature

Detected lane lines using masking and threshold technique
Perform a perspective transform using bird’s eye view
Fit a polynomial to lane lines
Detect lane lines and find curvature
One way to calculate the curvature of a lane line, is to fit a 2nd degree polynomial to that line, and from this you can easily extract useful information.
For a lane line that is close to vertical, you can fit a line using this formula: f(y) = Ay^2 + By + C, where A, B, and C are coefficients.
A gives you the curvature of the lane line, B gives you the heading or direction that the line is pointing, and C gives you the position of the line based on how far away it is from the very left of an image (y = 0).

2. Perspective transform:
Objects appear smaller the farther they are away
Parallel lines seems to converge to a point

Here we will get bird’s-eye view transform that lets us view a lane from above and which will be parallel and using this we can calculate lane curvature later on.

We will match 4 image points (src) on the road to desired image points (dst) on the perspective transformed image. Source and image points are described here:
Src:
Dst:

Note: I have chosen source points manually but I would like to explore in future a method which calculates these automatically.
  
Compute the perspective transform, M, given source and destination points:
M = cv2.getPerspectiveTransform(src, dst)
Warp an image using the perspective transform, M:
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

3. GRadient threshold
WE can use Canny edge detection but this gives lot of noise (tress, skies, sign boards and cars). For lane detection - we only need to consider lines which are vertical, for this we can use Sobel derivative of X as taking the gradient in the x-direction emphasizes edges closer to vertical.

Calculate the derivative in the x-direction (the 1, 0 at the end denotes x-direction):
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

Sample code:
Create a binary threshold to select pixels based on gradient strength:
thresh_min = 20
thresh_max = 100
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
plt.imshow(sxbinary, cmap='gray')

Direction of gradient:
So now we will explore the direction, or orientation, of the gradient.
The direction of the gradient is simply the arctangent of the y-gradient divided by the x-gradient. tan​−1​​(sobel​y​​/sobel​x​​). Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians, covering a range of −π/2 to π/2. An orientation of 0 implies a horizontal line and orientations of +/−π/2 imply vertical lines.
sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobelx=np.absolute(sobelx)
    abs_sobely=np.absolute(sobely)
    orient = np.arctan2(abs_sobely,abs_sobelx)
    binary_output=np.zeros_like(orient)
    binary_output[(orient>=thresh[0])&(orient<=thresh[1])]=1

HLS and Color Thresholds

We will use HSV color space to get valuable information about our lane lines - which are usually in white or yellow color. On the other hand, Lightness and Value represent different ways to measure the relative lightness or darkness of a color. For example, a dark red will have a similar hue but much lower value for lightness than a light red. Saturation also plays a part in this; saturation is a measurement of colorfulness. So, as colors get lighter and closer to white, they have a lower saturation value, whereas colors that are the most intense, like a bright primary color (imagine a bright red, blue, or yellow), have a high saturation value.
S channel does a robost job of picking up the lines under very different color and contrast conditions, under good conditions lane lines appear darker using H channel. Lets combine these 2 with a threshold to detect a better lane line. g how you might combine various color thresholds to make the most robust identification of the lines.

Yellow and white pixels

Combine threshold:
Now we can use various aspects of our gradient measurements (x, y, direction, color) to isolate lane-line pixels. Specifically, think about how you can use thresholds of the x and y gradients, the overall gradient magnitude, and the gradient direction to focus on pixels that are likely to be part of the lane lines.
At this point, it's okay to detect edges around trees or cars because these lines can be mostly filtered out by applying a mask to the image and essentially cropping out the area outside of the lane lines. It's most important that you reliably detect different colors of lane lines under varying degrees of daylight and shadow.
You can clearly see which parts of the lane lines were detected by the gradient threshold and which parts were detected by the color threshold by stacking the channels and seeing the individual components. You can create a binary combination of these two images to map out where either the color or gradient thresholds were met.



After doing these steps, you’ll be given two additional steps for the project:
Detect lane lines
Determine the lane curvature

Detect lane lines:
After applying calibration, thresholding, and a perspective transform to a road image, you should have a binary image where the lane lines stand out clearly. However, you still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.
I first take a histogram along all the columns in the lower half of the image like this:
import numpy as np
histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
plt.plot(histogram)
In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

In the last exercise, you located the lane line pixels, used their x and y pixel positions to fit a second order polynomial curve:
Skip the sliding windows step once you know where the lines are
Now you know where the lines are you have a fit! In the next frame of video you don't need to do a blind search again, but instead you can just search in a margin around the previous line position like this:


 If you lose track of the lines, go back to your sliding windows search or other method to rediscover them.

Curvature:
 You have a thresholded image, where you've estimated which pixels belong to the left and right lane lines (shown in red and blue, respectively, below), and you've fit a polynomial to those pixel positions. Next we'll compute the radius of curvature of the fit.

 our equation for radius of curvature becomes:
R​curve​​=​∣2A∣​​(1+(2Ay+B)​2​​)​3/2​​​​

leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                              for y in ploty])
rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                for y in ploty])


So we actually need to repeat this calculation after converting our x and y values to real world space.
This involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera, but for this project, you can assume that if you're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. Or, if you prefer to derive a conversion from pixel space to world space in your own images, compare your images with U.S. regulations that require a minimum lane width of 12 feet or 3.7 meters, and the dashed lane lines are 10 feet or 3 meters long each.


Finding your offset from lane center
You can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines you've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is your distance from the center of the lane.


Keep track:
ou're going to keep track of things like where your last several detections of the lane lines were, what the curvature was etc., so you can properly treat new detections. To do this, it's useful to define aLine() class to keep track of all the interesting parameters you measure from frame to frame. Here's an example:

If you lose track of the lines
If your sanity checks reveal that the lane lines you've detected are problematic for some reason, you can simply assume it was a bad or difficult frame of video, retain the previous positions from the frame prior and step to the next frame to search again. If you lose the lines for several frames in a row, you should probably go back to the blind search method using a histogram and sliding window, or other method, to re-establish your measurement.
Drawing the lines back down onto the road
Once you have a good measurement of the line positions in warped space, it's time to project your measurement back down onto the road! Let's suppose, as in the previous example, you have a warped binary image called warped, and you have fit the lines with a polynomial and have arrays called ploty, left_fitxand right_fitx, which represent the x and y pixel values of the lines. You can then project those lines onto the original image as follows:
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)











