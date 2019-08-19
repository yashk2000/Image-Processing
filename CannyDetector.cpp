#include <opencv2/opencv.hpp>

int main(int args, char** argv) {
    
    cv::Mat rgb, gry, cny;
    
    cv::namedWindow("Input Image(RGB)", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Grey Scale Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Canny Image", cv::WINDOW_AUTOSIZE);
    
    rgb = cv::imread(argv[1]);
    
    cv::imshow("Input Image(RGB)", rgb);
    
    cv::cvtColor(rgb, gry, cv::COLOR_BGR2GRAY);
    cv::imshow("Grey Scale Image", gry);
    
    cv::Canny(gry, cny, 10, 100, 3, true);
    cv::imshow("Canny Image", cny);
    
    cv::waitKey(0);
}
