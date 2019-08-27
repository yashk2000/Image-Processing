#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main(int args, char** argv) {
    
    cv::Mat img, result;
    img = cv::imread(argv[1]);
    
    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Histogram Equalized Image", cv::WINDOW_AUTOSIZE);
    
    if(img.empty()) {
        printf("No image path given as imput");
        return 1;
    }
    
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    
    cv::equalizeHist(img, result);
    
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    
    cv::imshow("Input Image", img);
    cv::imshow("Histogram Equalized Image", result);
    
    cv::waitKey(0);
    return 0;
}
    
    
