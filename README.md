# 1) Opening an image using openCV

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/openImage.cpp)

After compilation when the executable is run from the termilnal, provide a path to the place where the image is stored in the system. The image is displayed on the screen till the user presses any key.

Image is opened by the commnad `cv::imread(argv[1], -1);`. This can open an image of BMP, JPEG, PNG, PPM, RAS, etc formats and returns the image in a `cv::Mat` format. The image is opened by `cv::namedWindow`. It opens a window with the name given as a parameter to the function. This functions and the window which opens is provided by the **HighGUI Library**. The `WINDOW_AUTOSIZE` parameter automatically gives the window such a size as to accomodate the true size of the image. Now to display the image on the window, the `cv::imshow()` function is called. The `cv::waitKey()` function will paause the program execution and will wait until the user presses a key, at which point it will resume execution, hence calling the `cv::destroyWindow` function and closing the image.

Here's how the window with the photo will appear:(it might appear differently in different systems depending upon the photo)

![Screenshot_20190818_010130](https://user-images.githubusercontent.com/41234408/63216569-825c2000-c154-11e9-89c4-eab785f88094.png)

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

# 3) A simple transformation: Smoothing an image

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/imgSmoothing.cpp)

Here we will be smoothing an image using Gaussian Blur. Image smoothing basically refers to remove the sharp edges of an image. The resulting image is usually a bit more blurred that the input image. This tranformation is pretty simple. We take in an image as input in the same way as we did while opeing an image [earlier](https://github.com/yashk2000/Image-Processing#1-opening-an-image-using-opencv). After this, we create 2 windows, one for the initial image, and the other to display the smoothened image. We can smoothen the image by simply calling the `cv::GaussianBlur()` function. The Gaussain Blur function always takes in odd numbers as input. Here we have blurred the input image with a 5X5 Gaussian Blur convolution matrix. We have called the `cv::GaussianBlur()` function twice, so as to give us more pronounced effect. Here instead of Gaussian Blur, we can use other library methods such as `cv::blur()`, `cv::medianBlur()` or `bilateralFilter()`.

**Image before smoothing:**
![Screenshot_20190818_160716](https://user-images.githubusercontent.com/41234408/63223432-e1607a00-c1d2-11e9-9bb6-c4d4003b5ea6.png)

**Image after smoothing:**
![Screenshot_20190818_160734](https://user-images.githubusercontent.com/41234408/63223434-e4f40100-c1d2-11e9-9472-c5491f7c11d8.png)

