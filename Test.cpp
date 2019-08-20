#include <opencv2/opencv.hpp>

//this is just a random script in which I have combined both the pyrDown() and Canny() functions in a single pipeline.

int main(int args, char** argv) {
    
    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output Image", cv::WINDOW_AUTOSIZE);
    
    cv::Mat input, img_gry, img_pyr, img_pyr2, output;
    
    input = cv::imread(argv[1]);
    cv::imshow("Input Image", input);
    
    cv::cvtColor( input, img_gry, cv::COLOR_BGR2GRAY );
    cv::pyrDown( img_gry, img_pyr );
    cv::Canny( img_pyr, output, 10, 100, 3, true );

    cv::imshow("Output Image", output);
    
    cv::waitKey(0);
}
