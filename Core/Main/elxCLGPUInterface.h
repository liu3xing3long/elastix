#ifndef elxGLGPU_Resampler_h
#define elxGLGPU_Resampler_h

#include "itkImageSource.h"

// #include "elxElastixMain.h"
#include "elxParameterObject.h"
#include "elxPixelType.h"
#include "itkLightObject.h"
#include "itkGPUImage.h"

namespace elastix
{
///////////////////////////////////
// normal images
// cpu image
typedef float InterCPUPixType;
const unsigned int InterCPUDIM = 3;
typedef itk::Image< InterCPUPixType, InterCPUDIM >  CPUInputImageType;
typedef itk::Image< InterCPUPixType, InterCPUDIM >  CPUOutputImageType;

// gpu image
typedef float InterGPUPixType;
const unsigned int InterGPUDIM = 3;
typedef itk::GPUImage< InterGPUPixType, InterGPUDIM >  GPUInputImageType;
typedef itk::GPUImage< InterGPUPixType, InterGPUDIM >  GPUOutputImageType;

///////////////////////////////////
// binary images
// cpu image
typedef unsigned char InterBinCPUPixType;
typedef itk::Image< InterBinCPUPixType, InterCPUDIM >  BinCPUInputImageType;
typedef itk::Image< InterBinCPUPixType, InterCPUDIM >  BinCPUOutputImageType;

// gpu image
typedef unsigned char InterBinGPUPixType;
typedef itk::GPUImage< InterBinGPUPixType, InterGPUDIM >  BinGPUInputImageType;
typedef itk::GPUImage< InterBinGPUPixType, InterGPUDIM >  BinGPUOutputImageType;

/////////////////////////////////////
class ELASTIXLIB_API CLGPUInterface
{
public:
    CLGPUInterface();
    ~CLGPUInterface();

public:
    virtual void PrintInfo();
    GPUOutputImageType::Pointer GPUMemoryTest(CPUInputImageType::Pointer input);

public:
    bool IsGPUEnabled( void );

public:
    // resample floating images
    GPUOutputImageType::Pointer Resample(CPUInputImageType::Pointer input, std::vector<float> outSpacing);
    GPUOutputImageType::Pointer Threshold(CPUInputImageType::Pointer input, double lowerThreshold, double upperThreshold, double outsideValue);

    // morphing on binary images
    BinGPUOutputImageType::Pointer BinaryDilate(BinCPUInputImageType::Pointer input, int iRadius);
    BinGPUOutputImageType::Pointer BinaryErode(BinCPUInputImageType::Pointer input, int iRadius);
    BinGPUOutputImageType::Pointer BinaryThreshold(CPUInputImageType::Pointer input, double lowerThreshold, double upperThreshold, uint8_t insideValue, uint8_t outsideValue);

    // filtering functions
    GPUOutputImageType::Pointer Median(CPUInputImageType::Pointer inputImage, const std::vector<unsigned int>& radius);
    GPUOutputImageType::Pointer Mean(CPUInputImageType::Pointer inputImage, const std::vector<unsigned int>& radius);
    GPUOutputImageType::Pointer RecursiveGaussian(CPUInputImageType::Pointer inputImage, double sigma, bool normalizeAcrossScale, unsigned int order, unsigned int direction);
    GPUOutputImageType::Pointer DiscreteGaussian(CPUInputImageType::Pointer inputImage, double variance, unsigned int maximumKernelWidth, double maximumError, bool useImageSpacing);
    GPUOutputImageType::Pointer DiscreteGaussian(CPUInputImageType::Pointer inputImage, const std::vector< double > &variance, unsigned int maximumKernelWidth, const std::vector< double > &maximumError, bool useImageSpacing);
    GPUOutputImageType::Pointer GradientAnisotropicDiffusion(CPUInputImageType::Pointer inputImage, double timeStep, double conductanceParameter, unsigned int conductanceScalingUpdateInterval, uint32_t numberOfIterations);


public:
    void SetLastError(const char* err);
    const char* GetLastError();

private:
    // should call Init immediately after creating the interface before using any
    // real working methods
    bool Init( void );

private:
    char _last_error[4096];
};

}









#endif