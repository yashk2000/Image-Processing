#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int slider_pos = 0;
int run = 1, dontset = 0;
cv::VideoCapture cap;

void onTrackbarSlide(int pos, void *) {
    cap.set(cv::CAP_PROP_POS_FRAMES, pos);
    
    if (!dontset)
        run = 1;
    dontset = 0;
}

int main(int args, char** argv) {
    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    cap.open(string(argv[1]));
    int frames = (int) cap.get(cv::CAP_PROP_FRAME_COUNT);
    int w = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    
    cout << "Frames: " << frames << endl;
    cout << "Frame width: " << w << endl;
    cout << "Frame height: " << h << endl;
    
    cout << "Press r to run the video, s to pause or run the video frame by frame, h for help and esc to quit" << endl;
    
    cv::createTrackbar("Position", "Video", &slider_pos, frames, onTrackbarSlide);
    cv::Mat frame;
    
    while(true) {
        if(run != 0) {
            cap >> frame; 
            
            if(frame.empty())
                break;
            
            int current_position = (int)cap.get(cv::CAP_PROP_POS_FRAMES);
            dontset = 1;
            
            cv::setTrackbarPos("Position", "Video", current_position);
            cv::imshow("Video", frame);
            
            run = run - 1;
        }
        
        char ch = (char) cv::waitKey(10);
        
        if(ch == 's') {
            run = 1;
            cout << "Single Step, run = " << run << endl;
        }
        
        if(ch == 'r') {
            run = -1;
            cout << "Run mode, run = " << run << endl;
        }
        
        if(ch == 'h') {
            run = run;
            cout << "Press r to run the video, s to pause or run the video frame by frame, h for help and esc to quit" << endl;
        }
        
        if(ch == 27)
            break;
    }
    
    return(0);
}

