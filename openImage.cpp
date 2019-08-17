#include <opencv2/opencv.hpp>

int main (int args, char** argv) {
    
    cv::Mat img = cv::imread(argv[1], -1);
    if (img.empty()) {
        printf("No image found\n");
        return -1;
    }
    cv::namedWindow("Image1", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image1", img);
    cv::waitKey(0);
    cv::destroyWindow("Image1");
    return 0;
}
