# 1) Opening an image using openCV

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/openImage.cpp)

After compilation when the executable is run from the termilnal, provide a path to the place where the image is stored in the system. The image is displayed on the screen till the user presses any key.

Image is opened by the commnad `cv::imread(argv[1], -1);`. This can open an image of BMP, JPEG, PNG, PPM, RAS, etc formats and returns the image in a `cv::Mat` format. The image is opened by `cv::namedWindow`. It opens a window with the name given as a parameter to the function. This functions and the window which opens is provided by the **HighGUI Library**. The `WINDOW_AUTOSIZE` parameter automatically gives the window such a size as to accomodate the true size of the image. Now to display the image on the window, the `cv::imshow()` function is called. The `cv::waitKey()` function will paause the program execution and will wait until the user presses a key, at which point it will resume execution, hence calling the `cv::destroyWindow` function and closing the image.

# 2) Opening a video using openCV

The source code for this can be found [here](https://github.com/yashk2000/Image-Processing/blob/master/openVid.cpp)

The video can be opened and closed the same as the image in the previous section. 

The object `cap` of the type `cv::VideoCapture` is instatiated and is used to open or close videos. `cv::Mat` is used to create an object called `frame`. This object holds the frames of the video. The video is displayed frame by frame until there is no frame left which is checked by `frame.empty()`.
