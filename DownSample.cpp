#include <opencv2/opencv.hpp>

int main(int args, char** argv) {
    
    cv:: Mat input, output;
    
    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output Image", cv::WINDOW_AUTOSIZE);
    
    input = cv::imread(argv[1]);
    
    if(input.empty()) {
        printf("No input image given\n");
        return 0;
    }
    
    cv::imshow("Input Image", input);
    
    cv::pyrDown(input, output);
    
    cv::imshow("Output Image", output);
    
    cv::waitKey(0);
    
    return 0;
}
