#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>

using namespace std;

int main (int args, char** argv) {
    
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    cv::VideoCapture cap;
    cap.open(string(argv[1]));
    
    cv::Mat frame;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            printf("No frames left");
            break;
        }
        
        cv::imshow("Video", frame);
        
        if (cv::waitKey(33) >= 0) {
            printf("Exit");
            break;
        }
    }
    
    return 0;
}
