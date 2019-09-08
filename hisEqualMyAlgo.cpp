#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char** argv) {
    
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    
    int freq[256] = {0};
    for(int i = 0; i < img.rows; ++i) {
        for(int j = 0; j < img.cols; ++j) {
            freq[(int)img.at<uchar>(i, j)]++;
        }
    }
    
    int size = img.rows * img.cols;
    double sf = 255.0 / size;
    
    int cumfreq[256] = {0};
    cumfreq[0] = freq[0];
    for(int i = 1; i < 255; ++i) {
        cumfreq[i] = freq[i] + cumfreq[i-1];
    }
    
    int scale[256] = {0};
    for(int i = 0; i < 255; ++i) {
        scale[i] = cvRound((double)cumfreq[i] * sf);
    }
    
    cv::Mat equalizedImg = img.clone();
    
    for(int i = 0; i < img.rows; ++i) {
        for(int j = 0; j < img.cols; ++j) {
            equalizedImg.at<uchar>(i, j) = cv::saturate_cast<uchar>(scale[img.at<uchar>(i, j)]);
        }
    }
    
    cv::namedWindow("Original image in greyscale", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original image in greyscale", img);
    
    cv::namedWindow("Equalized image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Equalized image", equalizedImg);
    
    cv::waitKey(0);
    return 0;
}
            
    
