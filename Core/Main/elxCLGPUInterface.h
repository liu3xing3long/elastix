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
typedef itk::Image< InterCPUPixType, InterCPUDIM > CPUInputImageType;
typedef itk::Image< InterCPUPixType, InterCPUDIM > CPUOutputImageType;

// gpu image
typedef float InterGPUPixType;
const unsigned int InterGPUDIM = 3;
typedef itk::GPUImage< InterGPUPixType, InterGPUDIM > GPUInputImageType;
typedef itk::GPUImage< InterGPUPixType, InterGPUDIM > GPUOutputImageType;

///////////////////////////////////
// binary images
// cpu image
typedef unsigned char InterBinCPUPixType;
typedef itk::Image< InterBinCPUPixType, InterCPUDIM > BinCPUInputImageType;
typedef itk::Image< InterBinCPUPixType, InterCPUDIM > BinCPUOutputImageType;

// gpu image
typedef unsigned char InterBinGPUPixType;
typedef itk::GPUImage< InterBinGPUPixType, InterGPUDIM > BinGPUInputImageType;
typedef itk::GPUImage< InterBinGPUPixType, InterGPUDIM > BinGPUOutputImageType;

/////////////////////////////////////
class ELASTIXLIB_API CLGPUInterface
{
public:
    CLGPUInterface();
    
    ~CLGPUInterface();

public:
    virtual void
    PrintInfo();
    
    GPUOutputImageType::Pointer
    GPUMemoryTest( CPUInputImageType::Pointer input );

public:
    // should call Init immediately after creating the interface before using any
    // real working methods
    bool
    Init( std::vector< unsigned int > gpu_ids = std::vector< unsigned int >( 1, 0 ), bool bVerbose = false );

public:
    bool
    IsGPUAvailable( void );
    
    void
    PrintGPUInfo();

public:
    /// \brief resample given image at specified spacing
    /// \param input cpu itk image pointer
    /// \param outSpacing specified output image spacing
    /// \param uInterplolatorOrder interpolator type and order, 0 = NN, 1 = Linear, 2 and 3 = BSpline
    /// \param iDefault_voxel_value default voxel value outside the resample region
    /// \return resample image resampled image
    GPUOutputImageType::Pointer
    Resample( CPUInputImageType::Pointer input, std::vector< float > outSpacing = std::vector< float >( 3, 1.0 ),
              unsigned int uInterplolatorOrder = 3, int iDefault_voxel_value = -2048 );
    
    GPUOutputImageType::Pointer
    Threshold( CPUInputImageType::Pointer input, double lowerThreshold, double upperThreshold, double outsideValue );
    
    // morphing on binary images
    BinGPUOutputImageType::Pointer
    BinaryDilate( BinCPUInputImageType::Pointer input, int iRadius, unsigned int kernel_type, bool boundaryToForeground );
    BinGPUOutputImageType::Pointer
    BinaryDilate( BinCPUInputImageType::Pointer input, const std::vector< unsigned int > & vRadius, unsigned int kernel_type, bool boundaryToForeground );

    BinGPUOutputImageType::Pointer
    BinaryErode( BinCPUInputImageType::Pointer input, int iRadius, unsigned int kernel_type, bool boundaryToForeground );
    BinGPUOutputImageType::Pointer
    BinaryErode( BinCPUInputImageType::Pointer input, const std::vector< unsigned int > & vRadius, unsigned int kernel_type, bool boundaryToForeground );

    BinGPUOutputImageType::Pointer
    BinaryThreshold( CPUInputImageType::Pointer input, double lowerThreshold, double upperThreshold,
                     uint8_t insideValue, uint8_t outsideValue );
    
    // filtering functions
    GPUOutputImageType::Pointer
    Median( CPUInputImageType::Pointer inputImage, const std::vector< unsigned int > &radius );
    
    GPUOutputImageType::Pointer
    Mean( CPUInputImageType::Pointer inputImage, const std::vector< unsigned int > &radius );
    
    GPUOutputImageType::Pointer
    RecursiveGaussian( CPUInputImageType::Pointer inputImage, double sigma, bool normalizeAcrossScale,
                       unsigned int order, unsigned int direction );
    
    GPUOutputImageType::Pointer
    DiscreteGaussian( CPUInputImageType::Pointer inputImage, double variance, unsigned int maximumKernelWidth,
                      double maximumError, bool useImageSpacing );
    
    GPUOutputImageType::Pointer
    DiscreteGaussian( CPUInputImageType::Pointer inputImage, const std::vector< double > &variance,
                      unsigned int maximumKernelWidth, const std::vector< double > &maximumError,
                      bool useImageSpacing );
    
    GPUOutputImageType::Pointer
    GradientAnisotropicDiffusion( CPUInputImageType::Pointer inputImage, double timeStep, double conductanceParameter,
                                  unsigned int conductanceScalingUpdateInterval, uint32_t numberOfIterations );


public:
    void
    SetLastError( const char *err );
    
    const char *
    GetLastError();

private:
    char _last_error[4096];
    bool m_Verbose;
};

}


#endif