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

class ELASTIXLIB_API CLGPUInterface
{
    public:
        CLGPUInterface();
        ~CLGPUInterface();

    public:
        virtual void PrintInfo();
        GPUOutputImageType::Pointer GPUMemoryTest(CPUInputImageType::Pointer input);
        GPUOutputImageType::Pointer Resample(CPUInputImageType::Pointer input);
        
        void SetLastError(const char* err);
        const char* GetLastError(); 
    private:
        char _last_error[4096];
};

}









#endif