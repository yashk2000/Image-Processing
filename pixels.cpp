#include <opencv2/opencv.hpp>
#include <iostream>

int main(int args, char** argv) {
    
    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Downscaled Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("GreyScale Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Canny Image", cv::WINDOW_AUTOSIZE);
    
    cv::Mat input, gry, pyr, cny;
    
    input = cv::imread(argv[1]);
    cv::imshow("Input Image", input);

    cv::cvtColor(input, gry, cv::COLOR_BGR2GRAY);
    cv::pyrDown(gry, pyr);
    cv::Canny(pyr, cny, 10, 100, 3, true);
    
    cv::imshow("GreyScale Image", gry);
    cv::imshow("Downscaled Image", pyr);
    cv::imshow("Canny Image", cny);
    
    int x = 16, y = 32;
    cv::Vec3b intensity = input.at< cv::Vec3b >(y, x);
    
    uchar blue = intensity[0];
    uchar green = intensity[1];
    uchar red = intensity[2];
    
    std::cout << "At (x, y) = (" << x << ", " << y << "): (blue, green, red) = (" << (unsigned int)blue << ", " << (unsigned int)green << ", " << (unsigned int)red << ")" << std::endl;
    
    std::cout << "Gray pixel there is: " << (unsigned int)gry.at<uchar>(y, x) << std::endl;
    
    x = x/4;
    y = y/4;
    
    std::cout << "Downscaled pixel there is: " << (unsigned int)pyr.at<uchar>(y, x) << std::endl;
    
    cny.at<uchar>(x, y) = 128;
    
    cv::namedWindow("New Canny", cv::WINDOW_AUTOSIZE);
    cv::imshow("New Canny", cny);
    
    cv::waitKey(0);
    
    return 0;
}
