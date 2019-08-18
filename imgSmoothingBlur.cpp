#include <opencv2/opencv.hpp>

void imageSmooothing(const cv::Mat & image) {
    cv::namedWindow("Input image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Output image", cv::WINDOW_AUTOSIZE);
    
    cv::imshow("Input image", image);
    
    cv::Mat out;
    
    cv::blur(image, out, cv::Size(5, 5));
    cv::blur(out, out, cv::Size(5, 5));
    
    cv::imshow("Output image", out);
    
    cv::waitKey(0);
}

int main (int args, char** argv) {
    
    cv::Mat img = cv::imread(argv[1], -1);
    
    if (img.empty()) {
        printf("No image found\n");
        return -1;
    }
    
    imageSmooothing(img);
}
