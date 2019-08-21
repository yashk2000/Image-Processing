#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

int main(int args, char* op[]) {
    
    cv::namedWindow("Camera Check", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    
    cap.open(0);
        
    if (!cap.isOpened()) {
        std::cerr << "Coudln't open camera" << std::endl;
        return -1;
    }
    
    cv::Size size(
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );
    
    cv::VideoWriter writer;
    writer.open(op[1], CV_FOURCC('M', 'J', 'P', 'G'), 120, size);
    
    cv::Mat frame, logpolar_frame;
    
    while (true) {
        cap >> frame;
        
        if(frame.empty()) {
            break;
        }
        cv::imshow("Camera Input", frame);
        
        cv::logPolar(
            frame,
            logpolar_frame,
            cv::Point2f(
                frame.cols/2,
                frame.rows/2
            ),
            40,
            cv::WARP_FILL_OUTLIERS
        );
        
        cv::imshow("Output", logpolar_frame);
        writer << logpolar_frame;
        
        if (cv::waitKey(33) >= 0) {
            break;
        }
    }
    
    return 0;
}

