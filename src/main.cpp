
#include <iostream>
#include <opencv2/opencv.hpp>
#include "ssim.hpp"

int main(int argc, char** argv) {
    // Create test data
    std::cout << "Read image" << std::endl;
    cv::Mat testImage1 = cv::imread(argv[1]);
    cv::Mat testImage2 = cv::imread(argv[2]);

    if (!testImage1.data || !testImage2.data) {
        std::cout << "Can't read Image" << std::endl;
        return 1;
    }

    Ssim ssim = Ssim(testImage1.channels());
    cv::Scalar mssimScore = ssim.computeMeanSsim(testImage1, testImage2);
    std::cout << "mssim:";
    for(int i = 0; i < mssimScore.channels; i++){
        std::cout << "  channel " << i << ": " << mssimScore[i];
    }
    std::cout << std::endl;
    return 0;
}
