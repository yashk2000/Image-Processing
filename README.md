# Contents

1. [Opening an image using openCV](https://github.com/yashk2000/Image-Processing#1-opening-an-image-using-opencv)
2. [Opening a video using openCV](https://github.com/yashk2000/Image-Processing#2-opening-a-video-using-opencv)
3. [Creating a video player with play, pause and a seekbar](https://github.com/yashk2000/Image-Processing#3-creating-a-video-player-with-play-pause-and-a-seekbar)
4. [A simple transformation: Smoothing an image](https://github.com/yashk2000/Image-Processing#4-a-simple-transformation-smoothing-an-image)
5. [Downsampling an image](https://github.com/yashk2000/Image-Processing#5-downsampling-an-image)
6. [Simple way to detect edges of an image using Canny Edge Detector](https://github.com/yashk2000/Image-Processing#6-simple-way-to-detect-edges-of-an-image-using-canny-edge-detector)
7. [Taking input from device camera](https://github.com/yashk2000/Image-Processing#7-taking-input-from-device-camera)
8. [Getting and setting pixels in an image](https://github.com/yashk2000/Image-Processing#8-getting-and-setting-pixels-in-an-image)
9. [Reading a video and writing it to a file](https://github.com/yashk2000/Image-Processing#9-reading-a-video-and-writing-it-to-a-file)
10. [Reading and writing a video recorded from camera to a file](https://github.com/yashk2000/Image-Processing/blob/master/README.md#10-reading-and-writing-a-video-recorded-from-camera-to-a-file)
11. [Data Types in openCV](https://github.com/yashk2000/Image-Processing#11-data-types-used-in-opencv)

# 1) Opening an image using openCV

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/openImage.cpp)

After compilation when the executable is run from the termilnal, provide a path to the place where the image is stored in the system. The image is displayed on the screen till the user presses any key.

Image is opened by the commnad `cv::imread(argv[1], -1);`. This can open an image of BMP, JPEG, PNG, PPM, RAS, etc formats and returns the image in a `cv::Mat` format. The image is opened by `cv::namedWindow`. It opens a window with the name given as a parameter to the function. This functions and the window which opens is provided by the **HighGUI Library**. The `WINDOW_AUTOSIZE` parameter automatically gives the window such a size as to accomodate the true size of the image. Now to display the image on the window, the `cv::imshow()` function is called. The `cv::waitKey()` function will paause the program execution and will wait until the user presses a key, at which point it will resume execution, hence calling the `cv::destroyWindow` function and closing the image.

Here's how the window with the photo will appear:(it might appear differently in different systems depending upon the photo)

![Screenshot_20190818_160716](https://user-images.githubusercontent.com/41234408/63223432-e1607a00-c1d2-11e9-9bb6-c4d4003b5ea6.png)

# 2) Opening a video using openCV

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/openVid.cpp)

The video can be opened and closed the same as the image in the previous section. 

The object `cap` of the type `cv::VideoCapture` is instatiated and is used to open or close videos. `cv::Mat` is used to create an object called `frame`. This object holds the frames of the video. The video is displayed frame by frame until there is no frame left which is checked by `frame.empty()`.

Here's how the most basic video player with HighGUI will look:

![Screenshot_20190818_005927](https://user-images.githubusercontent.com/41234408/63216547-4f199100-c154-11e9-8b2b-2dd79dc6380d.png)

# 3) Creating a video player with play, pause and a seekbar 

The source code for creating a video player can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/vidPlayer.cpp)

Here we use global variables to keep track of the trackbar position and update in when needed, along with the `cv::VideoCapture` object as global. The `run` variable helps in keeping displaying frames as long as it isn't 0. We update the position of the seekbar with every frame being displayed. But this will put us into **single-step** mode whoch we don't want. To avoid this, we use the `dontset` variable to help us stay in the **run** mode. 

```cpp

void onTrackbarSlide(int pos, void *) {
  cap.set(cv::CAP_PROP_POS_FRAMES, pos);
  if( !dontset )
    run = 1;
  dontset = 0;
}
```
We need the number of frames in the video in order to calliberate the seekbar. This we get by using the `cap.get(cv::CAP_PROP_FRAMECOUNT`. Next, using thr number of frames, and also setting a variable to get the position of the seekbar, we use `cv::createTrackbar` to create the trackbar.

```cpp

createTrackbar("Position", "Video", &slider_position, frames, onTrackbarSlide);
```

While the video is running, we keep taking user input, inside the while loop. The character variable, `ch` is used for this. If user enters `r`, the video enters **run** mode and plays like nay normal video. On pressing `s`, we enter the **single-step** mode. If the video is in run mode and then we press `s`, the video gets paused. Hence the pause functionality can be implemented this way too. If the user presses `h`, we display the user with options of what can be done:
`Press r to run the video, s to pause or run the video frame by frame, h for help and esc to quit`.

This way, just using c++, we can use the HighGUI package provided in openCV to create our own video player.

This is how the video player looks with the seekbar: 

![Screenshot_20190818_010241](https://user-images.githubusercontent.com/41234408/63216703-fe576780-c156-11e9-9a88-19dbad8e4b42.png)

And here's how the terminal will be lookking like while you use `r` or `s` or `h` or `esc` to control the player:

![Screenshot_20190818_010249](https://user-images.githubusercontent.com/41234408/63216712-1b8c3600-c157-11e9-8128-7b1477c1d0a0.png)

# 4) A simple transformation: Smoothing an image

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/imgSmoothing.cpp)

Here we will be smoothing an image using Gaussian Blur. Image smoothing basically refers to remove the sharp edges of an image. The resulting image is usually a bit more blurred that the input image. This tranformation is pretty simple. We take in an image as input in the same way as we did while opeing an image [earlier](https://github.com/yashk2000/Image-Processing#1-opening-an-image-using-opencv). After this, we create 2 windows, one for the initial image, and the other to display the smoothened image. We can smoothen the image by simply calling the `cv::GaussianBlur()` function. The Gaussain Blur function always takes in odd numbers as input. Here we have blurred the input image with a 5X5 Gaussian Blur convolution matrix. We have called the `cv::GaussianBlur()` function twice, so as to give us more pronounced effect. Here instead of Gaussian Blur, we can use other library methods such as `cv::blur()`, `cv::medianBlur()` or `bilateralFilter()`.

**Image before smoothing:**
![Screenshot_20190818_160716](https://user-images.githubusercontent.com/41234408/63223432-e1607a00-c1d2-11e9-9bb6-c4d4003b5ea6.png)

**Image after smoothing:**
![Screenshot_20190818_160734](https://user-images.githubusercontent.com/41234408/63223434-e4f40100-c1d2-11e9-9472-c5491f7c11d8.png)

## Other methods for image smoothing

### blur()

We can use the `cv::blur()` method in [this](https://github.com/yashk2000/Image-Processing/blob/master/imgSmoothingBlur.cpp) way.

**Image after blurring**
![Screenshot_20190818_165436](https://user-images.githubusercontent.com/41234408/63223842-29829b00-c1d9-11e9-8d3f-4e02883a9a8a.png)

### medianBlur()

We can use the `cv::medianblur()` method in [this](https://github.com/yashk2000/Image-Processing/blob/master/imageSmoothingMedianBlur.cpp) way.

**Image after blurring**
![Screenshot_20190818_165500](https://user-images.githubusercontent.com/41234408/63223833-07891880-c1d9-11e9-8de9-1e37b4790da1.png)

### bilateralFilter()

We can use the `cv::bilateralFilter()` method in [this](https://github.com/yashk2000/Image-Processing/blob/master/imgSmoothingBilateralFilter.cpp) way.

In bilateralFilter(), we need to keep the input and output images differnet, else we get an error. 

Bilateral filter also takes a gaussian filter in space, but one more gaussian filter which is a function of pixel difference. Gaussian function of space make sure only nearby pixels are considered for blurring while gaussian function of intensity difference make sure only those pixels with similar intensity to central pixel is considered for blurring. So it preserves the edges since pixels at edges will have large intensity variation. The texture of the image is reduced, but the edges are conserved.

**Image after blurring**
![Screenshot_20190818_165158](https://user-images.githubusercontent.com/41234408/63223805-9e090a00-c1d8-11e9-8c8b-da43f3f75df0.png)

# 5) Downsampling an image 

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/DownSample.cpp)

This is one of the applications of Gaussian Blurring. We use Gaussian Blurring to downsample an image to change the sccale in which an image is viewed. We use the `cv::pyrDown()` function to do both Gaussian Blurring and downsampling simultaneously.

**Result of Downsampling an image:**
![Screenshot_20190820_001735](https://user-images.githubusercontent.com/41234408/63291423-1b1fa680-c2e1-11e9-8165-c07b8e5f352d.png)

The first image is the input image, which is larger. The second one is the output image which is half the size of the the input image. This shows that the image was downscaled by a margin of 2.

# 6) Simple way to detect edges of an image using Canny Edge Detector

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/CannyDetector.cpp)

To detect image edges, we use the Canny Edge Detector. To do this, we first convert the image to grey scale using `cv::cvtColor()`. This is done because Canny Egde Detector needs only a sigle channel to write to and a grey scale image is a single-channel image. Once the image is in grey scale, we can proceed with edge detection using `cv::Canny`. This might work even without converting the image to grey scale, but since Canny is a **single-channel** writing function, it is recommended to create a single-channel image i.e. a grey scale image.

**The grey scale and the final image with edges highlight will look like the following:**

**Input Image**
![Screenshot_20190820_005900](https://user-images.githubusercontent.com/41234408/63294985-41494480-c2e9-11e9-997f-89eb1337c1f3.png)

**Grey Scale Image**
![Screenshot_20190820_005909](https://user-images.githubusercontent.com/41234408/63295000-49a17f80-c2e9-11e9-9fad-1a1a676db1c4.png)

**Canny Image with prominent edges**
![Screenshot_20190820_005918](https://user-images.githubusercontent.com/41234408/63295015-51612400-c2e9-11e9-83ab-87cab8973566.png)

**Canny image when detecting edges without converting to grey scale**
![Screenshot_20190820_011012](https://user-images.githubusercontent.com/41234408/63295042-5920c880-c2e9-11e9-98da-a37f12e0e3de.png)

Here you can see that some details are missing. It also shows some edges which are actually not image edges. The Canny image in which edges were detected after converting to grey scale has more details. That's the reason for using a single-channel image while doing edge detection using Canny.

# 7) Taking input from device camera

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/InputFromCamera.cpp)

Here, if we provide a path as input, the video at the provided path will be opened. Otherwise, the camera will open and start taking a video. If there are no input arguments, we use the `VideoCapture` object to open the camera. If there is only one camera conected to the device, then we pass 0 as a parameter to the `VideoCapture` object. If there are multiple cameras connected, we can pass -1 as the parameter to open any one camera randomly. 
The video is displayed in the same way as a video stored in the computer. We use the same process as given in [this](https://github.com/yashk2000/Image-Processing/blob/master/openVid.cpp) code. 

Now while displaying the video, we can put any effect we want. It is displayed frame by frame. So we can treat each frame as an image and put any effect we want on the frame. For Example:

**Normal Code**
```cpp

while (true) {
  cap >> frame;
  if( frame.empty() ) break;
  cv::imshow( "Example3", frame );
  if( cv::waitKey(33) >= 0 ) break;
}
```

**Code with grey scale effect while displaying video**
```cpp

while (true) {
  cap >> frame;
  if( frame.empty() ) break;
  cv::cvtColor( frame, output_frame, cv::COLOR_BGR2GRAY);  
  cv::imshow( "Output", output_frame );
  if( cv::waitKey(33) >= 0 ) break;
}
```
This will display your video in greyscale format.

Here's how it looks while displaying a video:

**Camera Check**

![Screenshot_20190820_1732571](https://user-images.githubusercontent.com/41234408/63347080-06402300-c374-11e9-8766-d8279b7f4a20.png)

**Normal Video being recorded by camera is displayed**

![Screenshot_20190820_173257](https://user-images.githubusercontent.com/41234408/63347062-f4f71680-c373-11e9-8be7-5bcff601c798.png)

**Grey Scaled Video**

![Screenshot_20190820_180003](https://user-images.githubusercontent.com/41234408/63347331-7b135d00-c374-11e9-8fe6-413a83a906d2.png)

# 8) Getting and setting pixels in an image

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/pixels.cpp)

Here we use the Vec3b object to store the values of Red, green and blue colored pixels in a RGB image. Vec3b represnts a vector with 3 byte enteries, with each byte representing one color. 

If we are using an image with a single color, we can simply get the number of pixels of that color in the image by doing :
```cpp

(unsigned int)image.at<uchar>(y, x) //x, y represent pixel coordinates
```

To set a particular pixel of an image to a particular color, we simply access the pixel using it's coordinates and set it to the desired value. 
```cpp

image.at<uchar>(x, y) = 128;
```

# 9) Reading a video and writing it to a file

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/WriteFile.cpp)

This is essentailly just an extension to the [code](https://github.com/yashk2000/Image-Processing/blob/master/openVid.cpp) where we opened an existing video file from the local storage of the computer. Here we just use the `cv::VideoWriter` to write our desired frames to the file we want. We specify that we are writing a video in the common MJPG(motion jpeg) by using the `CV_FOURCC` codec. Here, we have to give the `writer` object some parameters. The fitst parameter is the output file path, the second one is the output file type, then comes the number of frames and finally the dimensions(size) of the output frame/frames.

# 10) Reading and writing a video recorded from camera to a file

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/ReadWriteLogPolarFromCamera.cpp)

This program is just an extension of the [one](https://github.com/yashk2000/Image-Processing/blob/master/InputFromCamera.cpp) where we took input from camera and [one](https://github.com/yashk2000/Image-Processing/blob/master/WriteFile.cpp) in which we wrote a video to another file. Here we just have to give the path to the output file when we run the code, instead of a path to the input file.

There's just one more creative thing we did here. Instead of directly writing the output to a file, which will be way easier, we first converetd the camera input to **log polar** form and wrote that to an output file. Log polar form is something like the way our eyes actually process what they see. We'll go into the details of the log polar form later on.

# 11) Data types used in openCV

Here we will be noticing many of the data types ending with 2d, 2i, 2f, 3d, 3i, 3b and so on. Here the number basically represents the dimension of the data stored and the characters indicate the following: `i for integer`, `f for float`, `d for double` and `b for unsigned character`.

### cv::Vec<>

The `cv::Vec<>` is used as a container for almost any type of data(including objects, pointers etc.) in cpp. But we mainly use it as a container for primitive data types such as int, float, char etc. 
We will not be using the `cv::Vec<>` template much. Instead we use the aliases that exist for some common instatiations of `cv::Vec<>`. Some of these are:
- `cv::Vec2i` - This will serve as a two element integer vector
- `cv::Vec3i` - This will serve as a three element integer vector
- `cv::Vec4d` - This will serve as a four element double precision floating point vector

The limitation of the cv::Vec class is that it is not very effecient for handling large arrays. For that we have the `cv::Mat` class(which we will learn about later on).

### cv::Matx<>

The `cv::Matx<>` class is  is a fixed matrix class which is highly effecient for dealing with 2 x 2, 3 x 3 and 4 x 4 matrix operations. Here too the only drawback is that we need to know the matrix dimensions before hand, and this is not efficient when it comes to handling large sized arrays. Again thr `cv::Mat` class(which wiil be discussed later on) comes to the rescue here. 

### cv::Point<>

This class is similar to the `cv::Vec<>` class. We can store 2 to 3 primitive data types using instances of `cv::Point<>`. Here we can call the data stored using the dot operator and class insatance itself.
For example if the instance is `point`. For a vector, we'll do `point[0]`, `point[1]` and so on. Instead in the `cv::Point<>` class, we simply do, `point.x`, `point.y` and so on. 
Same as the `cv::Vec<>` class, `cv::Point<>` class also contains some aliases. The most commonly used ones are: 
- `cv::Point2i`
- `cv::Point2f`
- `cv::Point2d`
- `cv::Point3i`
- `cv::Point3f`
- `cv::Point3d`

| Operation | Example | 
| --- | --- |
| Default Constructor | cv::point 2i p; |
|                     | cv::point 3f p; |
| Member access| p.x, p.y, p.z(if 3 dimensional) |
| Dot Product | float x = p1.dot(p2) |
| Double Precision Dot Product | double x = p1.ddot(p2) |
| Cross Product | p1.cross(p2) |
| To check if point p lies inside rectangle r | p.inside(r) |

### cv::Scalar<>

- Used for storing a four dimensional point.
- It aliased to store a four component vector with double precision components.  
- Data is called by objects and not the dot operator. 
- Directly derived from the `cv::Vec<>` class( `cv::Vec<double, 4`)

### cv::Size<>

- Has two members: height and width
- It is an alias for `cv:Size2i`
- To store floating point data, use `cv::Size2f`

### cv::Rect<>

- Has four members: height, width, x and y
- Alias to store the ineger form of a rectangle 

### cv::RotatedRect<>

- Store a rectangle which is not axis aligned
- Contains:

         - `cv::Point2f` object, called `center`
         
         - `cv::Size2f` object, called `size`
         
         - A float called `angle`
