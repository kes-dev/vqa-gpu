/**
 * Purpose: calculating ssim and mssim on gpu
 * Author:  jomehah
 */

#ifndef SSIM_HPP
#define SSIM_HPP

#include <vector>
#include <opencv2/opencv.hpp>

class Ssim {
private:
    // Size of images
    // Constants for SSIM
    double C1, C2;

    // Configurable window size
    unsigned int channel;
    unsigned int windowSize;
    double standardDev;

    std::vector<cv::cuda::GpuMat> input1Channel, input2Channel;
    cv::cuda::GpuMat input1, input2, inputTemp1, inputTemp2;
    cv::cuda::GpuMat input1Sq, input2Sq, input12, mu1, mu2,
        mu1Sq, mu2Sq, mu12, sigma1Sq, sigma2Sq, sigma12;
    std::vector<cv::cuda::GpuMat> ssimMap;
    cv::cuda::GpuMat sumBuffer;
    cv::Ptr<cv::cuda::Filter> gauss;

public:
    /**
     * Default constructor
     * Default ssim parameter:
     * K1 = 0.01, K2 = 0.03, windowSize = 11, standardDev = 1.5
     */
    Ssim(unsigned int channel);

    /**
     * Constructor for configurable K1 and K2
     */
    Ssim(unsigned int channel, double K1, double K2 , unsigned int bitDepth,
            unsigned int windowSize, double standardDev);

    /**
     * Destructor for Ssim
     * Release all gpu memory
     */
    ~Ssim();

    /**
     * Compute SSIM synchronously
     * This function will only calcuate until the step before averaging
     * use getSsimMap() to get the ssim map before averaging
     * use getMeanSsim() to get mean ssim over all channels
     */
    void computeSsim(const cv::Mat& input1, const cv::Mat& input2);

    /**
     * Compute SSIM asynchronously, does not call stream.waitForCompletion(),
     * so the caller may perform other async operations on ssimMap
     */
    void computeSsimAsync(const cv::Mat& input1, const cv::Mat& input2,
            cv::cuda::Stream& stream);

    /**
     * Compute MSSIM.  Synchronous.
     * @param inputImage1 first image to compare
     * @param inputImage2 sencond image to compare
     */
    cv::Scalar computeMeanSsim(const cv::Mat& input1, const cv::Mat& input2);

    /**
     * Get ssim map before average in gpu, does not download to system memory
     */
    std::vector<cv::cuda::GpuMat> getSsimMapGpu();

    /**
     * Get ssim map before averaging. vector size equals to images' number of channels,
     * download from gpu.
     */
    std::vector<cv::Mat> getSsimMap();


    /**
     * Get Mean SSIM value
     * This cannot be called right after computeSsimAsync(), as the stream.waitForComplete()
     * has not yet called.  The caller must ensures stream.waitForComplete() is called before
     * executing getMeanSsim()
     */
    cv::Scalar getMeanSsim();

};

#endif
