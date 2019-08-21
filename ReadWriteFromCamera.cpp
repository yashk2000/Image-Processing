#include <opencv2/opencv.hpp>
#include <iostream>

int main(int args, char* argv[]) {
    
    cv::namedWindow("Input", cv::WINDOW_AUTOSIZE);
    
    cv::VideoCapture capture(argv[1]);
    
    double fps = capture.get(cv::CAP_PROP_FPS);
    cv::Size size(
        (int)capture.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT)
    );
    
    cv::VideoWriter writer;
    writer.open(argv[2], CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
    
    cv::Mat bgr_frame;
    
    while(true) {
        
        capture >> bgr_frame;
        if(bgr_frame.empty()) {
            break;
        }
        
        cv::imshow("Input", bgr_frame);
        
        writer << bgr_frame;
        
        char ch = cv::waitKey(10);
        if(ch == 27) {
            break;
        }
    }
    
    capture.release();
}
