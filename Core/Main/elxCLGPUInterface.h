#ifndef elxGLGPU_Resampler_h
#define elxGLGPU_Resampler_h

#include "itkImageSource.h"

#include "elxElastixMain.h"
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
        GPUOutputImageType::Pointer Resample(CPUInputImageType::Pointer input,  std::vector<float> outSpacing);
        
        // morphing on binary images
        BinGPUOutputImageType::Pointer BinaryDilate(BinCPUInputImageType::Pointer input, int iRadius);
        BinGPUOutputImageType::Pointer BinaryErode(BinCPUInputImageType::Pointer input, int iRadius);

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