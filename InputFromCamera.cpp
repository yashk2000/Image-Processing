#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

int main(int args, char** argv) {
    
    cv::namedWindow("Camera Check", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    
    if (args == 1) {
        cap.open(0);
    } else {
        cap.open(argv[1]);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Coudln't open camera" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    
    while (true) {
        cap >> frame;
        
        if(frame.empty()) {
            break;
        }
        
        cv::imshow("Camera Input", frame);
        if (cv::waitKey(33) >= 0) {
            break;
        }
    }
    
    return 0;
}

