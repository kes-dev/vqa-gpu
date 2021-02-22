#include "ssim.hpp"

Ssim::Ssim(unsigned int channel): Ssim(channel, 0.01, 0.03, 8, 11, 1.5) {}

Ssim::Ssim(unsigned int channel, double K1, double K2, unsigned int bitDepth, unsigned int windowSize, double standardDev ) {
    this->channel = channel;
    this->C1 = pow(K1 * ((double) pow(2, bitDepth)), 2);
    this->C2 = pow(K2 * ((double) pow(2, bitDepth)), 2);
    this->windowSize = windowSize;
    this->standardDev = standardDev;

    // Pre allocate result array
    for (int i = 0; i < channel; i++) {
        this->ssimMap.push_back(cv::cuda::GpuMat());
    }

    this->gauss = cv::cuda::createGaussianFilter(
           CV_32F, -1, cv::Size(windowSize, windowSize), standardDev);
}

Ssim::~Ssim(){
    for (int i = 0; i < channel; i++) {
        this->ssimMap[i].~GpuMat();
    }

    this->ssimMap.clear();
    this->input1Channel.clear();
    this->input2Channel.clear();
    this->gauss->clear();
}

void Ssim::computeSsim(const cv::Mat& inputImage1, const cv::Mat& inputImage2) {
    cv::cuda::Stream stream;
    computeSsimAsync(inputImage1, inputImage2, stream);
    stream.waitForCompletion();
}

void Ssim::computeSsimAsync(const cv::Mat& inputImage1, const cv::Mat& inputImage2,
        cv::cuda::Stream& stream) {

    input1.upload(inputImage1, stream);
    input2.upload(inputImage2, stream);
    input1.convertTo(inputTemp1, CV_32F, stream);
    input2.convertTo(inputTemp2, CV_32F, stream);
    cv::cuda::split(inputTemp1, input1Channel, stream);
    cv::cuda::split(inputTemp2, input2Channel, stream);

    for( int i = 0; i < this->channel; i++ ) {
        cv::cuda::multiply(input1Channel[i], input1Channel[i], input1Sq, 1, -1, stream);
        cv::cuda::multiply(input2Channel[i], input2Channel[i], input2Sq, 1, -1, stream);
        cv::cuda::multiply(input1Channel[i], input2Channel[i], input12, 1, -1, stream);

        gauss->apply(input1Channel[i], mu1, stream);
        gauss->apply(input2Channel[i], mu2, stream);

        cv::cuda::multiply(mu1, mu1, mu1Sq, 1, -1, stream);
        cv::cuda::multiply(mu2, mu2, mu2Sq, 1, -1, stream);
        cv::cuda::multiply(mu1, mu2, mu12, 1, -1, stream);

        gauss->apply(input1Sq, input1Sq, stream);
        gauss->apply(input2Sq, input2Sq, stream);
        gauss->apply(input12, input12, stream);

        cv::cuda::subtract(input1Sq, mu1Sq, sigma1Sq, cv::cuda::GpuMat(), -1, stream);
        cv::cuda::subtract(input2Sq, mu2Sq, sigma2Sq, cv::cuda::GpuMat(), -1, stream);
        cv::cuda::subtract(input12, mu12, sigma12, cv::cuda::GpuMat(), -1, stream);

        cv::cuda::add(mu12, C1 / 2.0, input1Sq, cv::cuda::GpuMat(), -1, stream);
        cv::cuda::add(sigma12, C2 / 2.0, input2Sq, cv::cuda::GpuMat(), -1, stream);
        cv::cuda::multiply(input1Sq, input2Sq, input12, 4, -1, stream);

        cv::cuda::addWeighted(mu1Sq, 1, mu2Sq, 1, C1, input1Sq, -1, stream);
        cv::cuda::addWeighted(sigma1Sq, 1, sigma2Sq, 1, C2, input2Sq, -1, stream);
        cv::cuda::multiply(input1Sq, input2Sq, mu12, 1, -1, stream);
        cv::cuda::divide(input12, mu12, ssimMap[i], 1, -1, stream);
    }
}


cv::Scalar Ssim::computeMeanSsim(const cv::Mat& inputImage1, const cv::Mat& inputImage2) {
    computeSsim(inputImage1, inputImage2);
    return getMeanSsim();
}

std::vector<cv::cuda::GpuMat> Ssim::getSsimMapGpu() {
    return ssimMap;
}

std::vector<cv::Mat> Ssim::getSsimMap() {
    std::vector<cv::Mat> ssimMapInSysMemory;
    cv::Mat buffer;
    for (int i = 0; i < ssimMap.size(); i++) {
        ssimMap[i].download(buffer);
        ssimMapInSysMemory.push_back(buffer);
    }
    return ssimMapInSysMemory;
}
cv::Scalar Ssim::getMeanSsim() {
    cv::Scalar mssim;
    for (int i = 0; i < ssimMap.size(); i++) {
        cv::Scalar s = cv::cuda::sum(ssimMap[i], sumBuffer);
        mssim.val[i] = s.val[0] / (ssimMap[i].rows * ssimMap[i].cols);
    }
    return mssim;
}
