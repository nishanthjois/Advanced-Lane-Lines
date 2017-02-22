# Advanced-Lane-Lines

Advanced Lane Finding Project: Detect lane lines using computer vision techniques. 

The goals of this project are the following:

1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image ("bird's-eye view").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Let's start!

### 1. Camera calibration:
[Code for this section is present in `cam_calibrate(loc)` function in IPython notebook]

Distortion causes changes in objects appearance (size, shape and relative distance). The first step in analyzing camera images, is to undo this distortion so that we can get correct and useful information out of them. 

To correct distortion we will take pictures of known shapes (usually it will be a chessboard - because it has regular high contrast patterns) from the camera we are using and then compare with chessboard on a flat surface.

Here, we will use provided camera images of a chessboard and use cv2 functions to correct distortions.

#### Steps for camera calibration and image undistortion: 

  1. Take multiple images of a chessboard on a flat surface [Chessboard images are present in /camera_cal folder].

  2. Read in chessboard images with corners 9x6.

  3. Map coordinates of corners of 2D image points to real 3D object points.

  4. Detect corners using `cv2.findChessboardCorners` - which returns corners found in a gray scale image.

  5. Append corners returned in previous step to image points array.

  6. Use `cv2.calibrateCamera` to calibrate camera.

    `ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)`
  
  7. Use distortion coefficients and camera matrix returned from previous step along with `cv2.undistort` to return an undistorted image.

  8. Use these distortion coefficients and camera matrix to undistort every frame of video in the pipeline; see below for sample:

![Alt text](/Output-images/Chessboardcorners.png?)

![Alt text](/Output-images/distored.png?)

![Alt text](/Output-images/undistored.png?)


### 2. Perspective transform:
[Code for this section is present in pipeline() and warp(img) function in IPython notebook; code is also described below]

Objects appear smaller the farther they are away and parallel lines seems to converge to a point hence we perform perspective transform to fix these issues.

For this project, we take a frame of the road and transform it to bird’s-eye view that lets us view a lane from above. In this view, lanes will be parallel and using this we can calculate lane curvature later on.

We will match 4 image points (src) on the road to desired image points (dst) on the perspective transformed image. Source and image points are described here:

   
    src = np.array([[490, 482],[810, 482], [1250, 720],[40, 720]], dtype=np.float32)
   
    dst = np.array([[0, 0], [1280, 0], [1250, 720],[40, 720]], dtype=np.float32)

Note: I have chosen source points manually but I would like to explore in future a method which calculates these automatically.
  
Compute the perspective transform, M, given source and destination points using:

  `
  M = cv2.getPerspectiveTransform(src, dst)`
  
Warp an image using the perspective transform, M:

  `
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)`

This is result:

![Alt text](/Output-images/transform.png?)

### 3. Gradient threshold
[Code for this section is present in `color_gradient_threshold(img)` function in IPython notebook]

Now we will have to detect lines of a lane in the frame, we can use Canny edge detection but this gives lot of noise (tress, skies, sign boards and cars). 

For lane detection - we only need to consider lines which are vertical, for this we can use Sobel derivative of X; as taking the gradient in the x-direction emphasizes edges closer to vertical.

Calculate the derivative in the x-direction (the 1, 0 at the end denotes x-direction):

  `
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)`

Sample code:
Create a binary threshold to select pixels based on gradient strength:

  
  
    thresh_min = 20
 
    thresh_max = 100
  
    sxbinary = np.zeros_like(scaled_sobel)
  
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
  
    plt.imshow(sxbinary, cmap='gray')

![Alt text](/Output-images/sobelx.png?)

### 4. Direction of gradient:
[Code for this section is present in `color_gradient_threshold(img)` function in IPython notebook]

Now we will explore the direction, or orientation, of the gradient. 

The direction of the gradient is simply the arctangent of the y-gradient divided by the x-gradient. Each pixel of the resulting image contains a value for the angle of the gradient away from horizontal in units of radians, covering a range of −π/2 to π/2. An orientation of 0 implies a horizontal line and orientations of +/−π/2 imply vertical lines.

   
    sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
   
    sobely=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    
    abs_sobelx=np.absolute(sobelx)
    
    abs_sobely=np.absolute(sobely)
    
    orient = np.arctan2(abs_sobely,abs_sobelx)
    
    binary_output=np.zeros_like(orient)
    
    binary_output[(orient>=thresh[0])&(orient<=thresh[1])]=1
    

![Alt text](/Output-images/direction_gradient.png?)

#### 5. HLS Thresholds
[Code for this section is present in `color_gradient_threshold(img)` function in IPython notebook]

We will use HSV color space to get valuable information about our lane lines. Hue is the term for the pure spectrum colors commonly referred to by the 'color names'.'Value' represent different ways to measure the relative lightness or darkness of a color. For example, a dark red will have a similar hue but much lower value for lightness than a light red. Saturation also plays a part in this; saturation is a measurement of colorfulness. So, as colors get lighter and closer to white, they have a lower saturation value, whereas colors that are the most intense, like a bright primary color (imagine a bright red, blue, or yellow), have a high saturation value.

As per our experiments (and class tutorials) 'S' channel does a robust job of picking up the lines under very different color and contrast conditions. Under good conditions lane lines appear darker using H channel. Lets combine these 2 with a threshold to detect a better lane line.

   
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    hue = hls[:, :, 0]
    lightness = hls[:, :, 1]
    saturation = hls[:, :, 2]
    
    # Threshold s channel
    saturation_threshold=(170, 255)
    s_binary = np.zeros_like(saturation)
    s_binary[(saturation >= saturation_threshold[0]) & (saturation <= saturation_threshold[1])] = 1
    
    # Threshold h channel
    hue_threshold=(20, 100)
    h_binary = np.zeros_like(hue)
    h_binary[(hue >= hue_threshold[0]) & (hue <= hue_threshold[1])] = 1

![Alt text](/Output-images/huechannel.png?)

![Alt text](/Output-images/schannel.png?)

#### 6. Yellow and white pixels
[Code for this section is present in `color_gradient_threshold(img)` function in IPython notebook]

Previous transformations were all in gray scale due to this we lose valuable information about colors. It's important that we reliably detect different colors of lane lines under varying degrees of daylight and shadow. Lane lines are typically in yellow and white colors so let's combine various color thresholds to make the most robust identification of the lines.

   
    yellow_lane= cv2.inRange(img, (200,200,0), (255,255,150))
   
    white_lane= cv2.inRange(img, (200, 200, 200), (255, 255, 255))
    
    yellow_and_white_img = yellow_lane | white_lane
    
    yellow_and_white_img = np.divide(yellow_and_white_img, 255)
    

![Alt text](/Output-images/yellow_white.png?)

#### 7. Combine threshold:
[Code for this section is present in `color_gradient_threshold(img)` function in IPython notebook]

Now we can use various aspects of our gradient measurements (x, y, direction, color) to isolate lane-line pixels. 

At this point, it's okay to detect edges around trees or cars because these lines can be mostly filtered out by applying a mask to the image and essentially cropping out the area outside of the lane lines. Finally we will create a binary combination of all these thresholded parameters. 

![Alt text](/Output-images/combined_thresholds.png?)

#### 8. Detect lane lines using histogram and sliding window:
[Code for this section is present in `detect_first_lane(binary_warped)` function in IPython notebook]

##### Step #1: 

After applying calibration, thresholding, and a perspective transform to a road image, we should have a binary image where the lane lines stand out clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

We first take a histogram along all the columns in the lower half of the image like this:
    
    
      histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

![Alt text](/Output-images/histogram.png?)

##### Step #2: 

In thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. 
   
    
     # Find the peak of the left and right halves of the histogram.
    These will be the starting point for the left and right lines
  
    midpoint = np.int(histogram.shape[0]/2)
    
    leftx_base = np.argmax(histogram[:midpoint])
    
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint  
    
    
##### Step #3: 

We can use above mentioned x-position as a starting point for where to search for the lines. From that point, we can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

This section we iterate nwindows times and in each iteration we identify window's 4 sides, find good pixels for right and left lanes and append to lane indices for further processing. Now we find lane line pixels, use their x and y pixel positions to fit a second order polynomial curve; we also re-center next window if minimum number of pixels are found :

Note: Polynomial is calculated using `x = ay^2 + by + c`

    # Step through the windows one by one
    for window in range(nwindows):
       # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
           leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
           rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    

![Alt text](/Output-images/lanelines.png?)

##### Step #4:

[Code for this section is present in `detect_next_lane(binary_warped, left_fit, right_fit)` function in IPython notebook]

We will skip the sliding windows step once we know where the lines, in the next frame of video we don't need to do a blind search again, but instead we can just search in a margin around the previous line position like this:

![Alt text](/Output-images/laneplotted.png?)

##### Step #5: 

[Code for this section is present in `sanity_check(lane, radius_l, radius_r)` function in IPython notebook]

We will do a sanity check to check if we have lost track of the lane lines due to bad or difficult frame of video.

If sanity check fails we retain the previous positions from the frame prior and step to the next frame to search again. If we lose the lines for several frames in a row, we go back to the blind search method using a histogram and sliding window to re-establish your measurement.

#### 9. Curvature:
[Code for this section is present in `radius_of_curvature(image, leftx, rightx, ploty, l, r)` function in IPython notebook]

We have a thresholded image, where we've estimated which pixels belong to the left and right lane lines and we've fit a polynomial to those pixel positions. Next we'll compute the radius of curvature of the fit.

Note: Radius of curavture is calculated using
`curvature = pow(1 + (2*a*y + b)**2, 1.5) / math.fabs (2*a)`
 
Our equation for radius of curvature becomes:
  
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

So we actually need to repeat this calculation after converting our x and y values to real world space.

This involves measuring how long and wide the section of lane is that we're projecting in our warped image. For this project, we are assuming that the lane is about 30 meters long and 3.7 meters wide. 

  
    # define conversion in x and y from pixel space to meters
    y_eval = 719
    
    ym_per_pix = 30/720
    
    xm_per_pix = 3.7/(l-r)
    

### 10. Finding your offset from lane center

We can assume the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines you've detected. The offset of the lane center from the center of the image (converted from pixels to meters) is your distance from the center of the lane.

  
    Calculate lane positions 
  
    mid_lane = (np.max(fit_rightx) - np.min(fit_leftx))
    
    left = np.min(fit_leftx)
    
    right = np.max(fit_rightx)


### 11. Pipepline

We will combine all our functions mentioned above and create a pipeline. This pipeline takes one frame at a time and returns warped image.

### 12. Drawing the lines back down onto the road

[Code for this section is present in `lanes_warped (warped, left_fitx, right_fitx, ploty)` function in IPython notebook]

Once we have a good measurement of the line positions in warped space, it's time to project our measurement back down onto the road! 

We have a warped binary image called warped, and we have fit the lines with a polynomial and have arrays called ploty, left_fitx and right_fitx, which represent the x and y pixel values of the lines. We can then project those lines onto the original image as follows:

    
    # Create an image to draw the lines on
    
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    Recast the x and y points into usable format for cv2.fillPoly()
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    pts = np.hstack((pts_left, pts_right))

    #Draw the lane onto the warped blank image
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    #Warp the blank back to original image space using inverse perspective matrix (Minv)
    
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    
    Combine the result with the original image
    
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    plt.imshow(result)

![Alt text](/Output-images/detectedlanemerged.png?)

### 13. Store camera calibration file

Instead of calibrating our input frame every time, we calibrate and store the camera calibration file and load whenever required.

    
    #If camera has not been calibrated
    
    if not os.path.isfile(calibrated_file):
    
       cam_matrix, dist_coeff = cam_calibrate('./camera_cal/calibration*.jpg')
       
       np.savez_compressed(calibrated_file, cam_matrix=cam_matrix, dist_coeff=dist_coeff)
       
    else:
    
       # Camera has been already calibrated
       
       data = np.load(calibrated_file)
       
       cam_matrix = data['cam_matrix']
       
       dist_coeff = data['dist_coeff']
       

### 14. Read and save output
These 2 lines are declared in our main function (which also contains a call to pipeline()) to read test file and write the video to result

    
    # Read input file
    
    input_file = VideoFileClip('./project_video.mp4')
    
    # Store input file
    
    result.write_videofile(output_file , audio=False)

![Alt text](/Output-images/outputframe.png?)

#### Output video:

[![IMAGE ALT TEXT](http://img.youtube.com/vi/nnbsHtPdJFE/0.jpg)](http://www.youtube.com/watch?v=nnbsHtPdJFE "Output video")

#### Issues I faced:

1. Had issue in getting perspective transform.

2. Was not sure how to do sanity check.

3. Had trouble understanding how plotting and ployfit works (my mentor helped me understand this with diagrams).

4. There were lot of minute tweaks - which  I had to do to get this working.

For all the issues I faced, mentor, forums and slack channels came to rescue, thanks! 

#### Future work:

1. Make the code work on advanced challenges

2. Do not hardcode src, dst points for perspective transform

3. Use low pass filter to smoothen lane detection
