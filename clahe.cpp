#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

int main(int argc, char** argv) {
    
    Mat m= imread("/home/yashk2000/Downloads/beach.jpeg",IMREAD_GRAYSCALE); 
imshow("lena_GRAYSCALE",m);

Mat result;
Ptr<CLAHE> clahe = createCLAHE();
clahe->setClipLimit(10);
equalizeHist(m, result);
Mat dst;
clahe->apply(m,dst);
imshow("lena_CLAHE",dst);
imshow("Histogram", result);

waitKey();
}
