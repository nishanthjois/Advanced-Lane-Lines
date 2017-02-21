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
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

Distortion causes changes in objects appearance (size, shape and relative distance). The first step in analyzing camera images, is to undo this distortion so that we can get correct and useful information out of them. 

To correct distortion we will take pictures of known shapes (usually it will be a chessboard - because it has regular high contrast patterns) from the camera we are using and compare with chessboard on a flat surface.

Here, we will use provided camera images of a chessboard and use cv2 functions to correct distortions.


#### Steps for camera calibration and image undistortion: 

1. Take multiple images of a chessboard on a flat surface [Chessboard images are present in /camera_cal folder].

2. Read in chessboard images with corners 9x6.

3. Map coordinates of corners of 2D image points to real 3D object points.

4. Detect corners using `cv2.findChessboardCorners` - which retruns corners found in a gray scale image.

![Alt text](/Output-images/Chessboardcorners.png?)

5. Append corners returned in previous step to image points array.

6. Use `cv2.calibrateCamera` to calibrate camera.

`ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)`
  
7. Use distortion coefficients and camera matrix returned from previous step along with `cv2.undistort` to return an undistorted image.

8. Use these distortion coefficients and camera matrix to undistort every frame of video in the pipeline; see below for sample:

![Alt text](/Output-images/distored.png?)

![Alt text](/Output-images/undistored.png?)














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
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

Objects appear smaller the farther they are away and parallel lines seems to converge to a point hence we perform perspective transform to fix these issue.

For this project, we take a frame of the road and transform it to bird’s-eye view that lets us view a lane from above. In this view lanes will be parallel and using this we can calculate lane curvature later on.

We will match 4 image points (src) on the road to desired image points (dst) on the perspective transformed image. Source and image points are described here:
Src:
Dst:

Note: I have chosen source points manually but I would like to explore in future a method which calculates these automatically.
  
Compute the perspective transform, M, given source and destination points using:
`M = cv2.getPerspectiveTransform(src, dst)`
Warp an image using the perspective transform, M:
`warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)`


#### 3. Gradient threshold
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

Now we will have to detect lines of a lane in the frame, we can use Canny edge detection but this gives lot of noise (tress, skies, sign boards and cars). For lane detection - we only need to consider lines which are vertical, for this we can use Sobel derivative of X; as taking the gradient in the x-direction emphasizes edges closer to vertical.

Calculate the derivative in the x-direction (the 1, 0 at the end denotes x-direction):
`sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)`

Sample code:
Create a binary threshold to select pixels based on gradient strength:
thresh_min = 20
thresh_max = 100
`sxbinary = np.zeros_like(scaled_sobel)`
`sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1`
`plt.imshow(sxbinary, cmap='gray')`

![Alt text](/Output-images/sobelx.png?)

#### Direction of gradient:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

Now we will explore the direction, or orientation, of the gradient. The direction of the gradient is simply the arctangent of the y-gradient divided by the x-gradient. Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians, covering a range of −π/2 to π/2. An orientation of 0 implies a horizontal line and orientations of +/−π/2 imply vertical lines.
   `sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobelx=np.absolute(sobelx)
    abs_sobely=np.absolute(sobely)
    orient = np.arctan2(abs_sobely,abs_sobelx)
    binary_output=np.zeros_like(orient)
    binary_output[(orient>=thresh[0])&(orient<=thresh[1])]=1`

![Alt text](/Output-images/direction_gradient.png?)

#### HLS and Color Thresholds
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]
We will use HSV color space to get valuable information about our lane lines. Hue is the term for the pure spectrum colors commonly referred to by the 'color names'.'Value' represent different ways to measure the relative lightness or darkness of a color. For example, a dark red will have a similar hue but much lower value for lightness than a light red. Saturation also plays a part in this; saturation is a measurement of colorfulness. So, as colors get lighter and closer to white, they have a lower saturation value, whereas colors that are the most intense, like a bright primary color (imagine a bright red, blue, or yellow), have a high saturation value.

As per our experiments (and class tutorials) S channel does a robost job of picking up the lines under very different color and contrast conditions. Under good conditions lane lines appear darker using H channel. Lets combine these 2 with a threshold to detect a better lane line.

![Alt text](/Output-images/huechannel.png?)

![Alt text](/Output-images/schannel.png?)

#### Yellow and white pixels
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]
Lane lines are typically in yellow and white colors so lets combine various color thresholds to make the most robust identification of the lines.

![Alt text](/Output-images/yellow_white.png?)

#### Combine threshold:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]
Now we can use various aspects of our gradient measurements (x, y, direction, color) to isolate lane-line pixels. 
At this point, it's okay to detect edges around trees or cars because these lines can be mostly filtered out by applying a mask to the image and essentially cropping out the area outside of the lane lines. It's most important that we reliably detect different colors of lane lines under varying degrees of daylight and shadow.
We can clearly see which parts of the lane lines were detected by the gradient threshold and which parts were detected by the color threshold by stacking the channels and seeing the individual components. Finally we will create a binary combination of all these thresholded parameters. 

![Alt text](/Output-images/combined_thresholds.png?)

After doing these steps, we have 2 more steps to go:
Detect lane lines
Determine the lane curvature

#### Detect lane lines using histogram and sliding window:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

Step #1: 
After applying calibration, thresholding, and a perspective transform to a road image, we should have a binary image where the lane lines stand out clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.
We first take a histogram along all the columns in the lower half of the image like this:
`import numpy as np
histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
plt.plot(histogram)`

![Alt text](/Output-images/histogram.png?)

Step #2: 
In thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. 

Step #3: 
We can use above mentioned x-position as a starting point for where to search for the lines. From that point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.
Now we find lane line pixels, use their x and y pixel positions to fit a second order polynomial curve:

`code`

![Alt text](/Output-images/lanelines.png?)

Step #4:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

We will skip the sliding windows step once we know where the lines, in the next frame of video we don't need to do a blind search again, but instead we can just search in a margin around the previous line position like this:
`code`

![Alt text](/Output-images/laneplotted.png?)

Step #5:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

We will do a sanity check to check if we have lost track of the lane lines due to bad or difficult frame of video.
If sanity check fails we retain the previous positions from the frame prior and step to the next frame to search again. If we lose the lines for several frames in a row, we go back to the blind search method using a histogram and sliding window to re-establish your measurement.

#### Curvature:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

We have a thresholded image, where we've estimated which pixels belong to the left and right lane lines and we've fit a polynomial to those pixel positions. Next we'll compute the radius of curvature of the fit.

Our equation for radius of curvature becomes:
R​curve​​=​∣2A∣​​(1+(2Ay+B)​2​​)​3/2​​​​

`leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])
rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) for y in ploty])`

So we actually need to repeat this calculation after converting our x and y values to real world space.
This involves measuring how long and wide the section of lane is that we're projecting in our warped image. We could do this in detail by measuring out the physical lane in the field of view of the camera, but for this project, you can assume that if you're projecting a section of lane similar to the images above, the lane is about 30 meters long and 3.7 meters wide. 


### Finding your offset from lane center
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

We can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines you've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is your distance from the center of the lane.


#### Pipepline
We will combine all our functions mentioned above and create a pipeline. This pipeline takes one frame at a time and returns wraped image

#### Drawing the lines back down onto the road
Once you have a good measurement of the line positions in warped space, it's time to project our measurement back down onto the road! We have a warped binary image called warped, and we have fit the lines with a polynomial and have arrays called ploty, left_fitx and right_fitx, which represent the x and y pixel values of the lines. We can then project those lines onto the original image as follows:

`Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
plt.imshow(result)`

![Alt text](/Output-images/detectedlanemerged.png?)

#### Keep track:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

We're going to keep track of things like where your last several detections of the lane lines were, what the curvature was etc., so you can properly treat new detections. To do this, it's useful to define aLine() class to keep track of all the interesting parameters you measure from frame to frame. Here's an example:


#### Read and save output

![Alt text](/Output-images/outputframe.png?)















