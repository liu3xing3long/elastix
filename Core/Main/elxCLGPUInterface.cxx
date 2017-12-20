
// Other include files
#include <iomanip> // setprecision, etc.
#include <sstream>

#include "itkCLGPUInterfaceHelper.h"

// GPU include files
#include "itkGPUResampleImageFilter.h"

// ITK include files
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkOutputWindow.h"
#include "itkTimeProbe.h"

#include "elxCLGPUInterface.h"
#include "itkCLGPUInterfaceHelper.h"

namespace elastix
{

//------------------------------------------------------------------------------
GPUOutputImageType::Pointer CLGPUInterface::GPUMemoryTest(CPUInputImageType::Pointer itkImage)
{
  GPUInputImageType::Pointer itkGPUImage = GPUInputImageType::New();
#ifndef NDEBUG
  std::cout << "begin" << std::endl;
#endif
  itkGPUImage->Print(std::cout);

#ifndef NDEBUG
  std::cout << "after graft" << std::endl;
#endif
  itkGPUImage->GraftITKImage(itkImage);
  itkGPUImage->Print(std::cout);
  
#ifndef NDEBUG
  std::cout << "after allocate" << std::endl;
#endif
  itkGPUImage->AllocateGPU();
  itkGPUImage->Print(std::cout);

#ifndef NDEBUG
  std::cout << "after update buffer" << std::endl;
#endif
  itkGPUImage->GetGPUDataManager()->SetCPUBufferLock( true );
  itkGPUImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
  itkGPUImage->GetGPUDataManager()->UpdateGPUBuffer();

  itkGPUImage->Print(std::cout);
  
  return itkGPUImage;
}

//------------------------------------------------------------------------------
void CLGPUInterface::PrintInfo()
{

}


//------------------------------------------------------------------------------
GPUOutputImageType::Pointer CLGPUInterface::Resample(CPUInputImageType::Pointer itkImage)
{
  // basic typedefs
  typedef CPUInputImageType::SizeType::SizeValueType  SizeValueType;
  typedef typelist::MakeTypeList< float >::Type    OCLImageTypes;
  typedef float InterpolatorPrecisionType;
  typedef float ScalarType;
 
  typedef itk::GPUResampleImageFilter < GPUInputImageType, GPUOutputImageType, InterpolatorPrecisionType > FilterType;
  
  //CPU interpolator here
  typedef itk::InterpolateImageFunction< CPUInputImageType, InterpolatorPrecisionType >             InterpolatorType;
  typedef itk::GPUInterpolatorCopier< OCLImageTypes, OCLImageDims, InterpolatorType, InterpolatorPrecisionType > InterpolateCopierType;
  
  // Transform typedefs
  typedef itk::Transform< ScalarType, InterCPUDIM, InterCPUDIM > TransformType;
  typedef itk::TranslationTransform< ScalarType, InterCPUDIM > TranslationTransformType;
  typedef itk::GPUTransformCopier< OCLImageTypes, OCLImageDims, TransformType, ScalarType > TransformCopierType;

  // CPU part
  InterpolatorType::Pointer cpuInterpolator;
  const unsigned int splineOrderInterpolator = 3;
  itk::DefineInterpolator< InterpolatorType >( cpuInterpolator, "BSpline", splineOrderInterpolator );
  
  // Create Translation transform
  TransformType::Pointer    cpuTransform;
  typename TranslationTransformType::Pointer translationTransform = TranslationTransformType::New();
  typename TranslationTransformType::ParametersType parameters;
  itk::DefineTranslationParameters< TranslationTransformType > ( 0, parameters );
  translationTransform->SetParameters( parameters );
  cpuTransform = translationTransform;

  // obtain original params
  const CPUInputImageType::SpacingType   inputSpacing   = itkImage->GetSpacing();
#ifndef NDEBUG
  std::cout << "obtaining params." << std::endl;
#endif
  const CPUInputImageType::PointType     inputOrigin    = itkImage->GetOrigin();
  const CPUInputImageType::DirectionType inputDirection = itkImage->GetDirection();
  const CPUInputImageType::RegionType    inputRegion    = itkImage->GetBufferedRegion();
  const CPUInputImageType::SizeType      inputSize      = inputRegion.GetSize();
#ifndef NDEBUG
  std::cout << "input params obtained." << std::endl;
#endif
  CPUOutputImageType::SpacingType   outputSpacing;
  CPUOutputImageType::PointType     outputOrigin;
  CPUOutputImageType::DirectionType outputDirection;
  CPUOutputImageType::SizeType      outputSize;
  double                         tmp1, tmp2;
  std::stringstream              s; 
  s << std::setprecision( 4 ) << std::setiosflags( std::ios_base::fixed );

  for( std::size_t i = 0; i < InterCPUDIM; i++ )
  {
    tmp1 = 1.0 / inputSpacing[i];
    tmp2 = inputSpacing[ i ] * tmp1;
    s << tmp2; s >> outputSpacing[ i ]; s.clear();

    tmp2 = inputOrigin[ i ];
    s << tmp2; s >> outputOrigin[ i ]; s.clear();

    for( unsigned int j = 0; j < InterCPUDIM; j++ )
    {
      outputDirection[ i ][ j ] = inputDirection[ i ][ j ];        // * tmp;
    }
    outputSize[ i ] = itk::Math::Round< SizeValueType >( inputSize[ i ] * inputSpacing[i] / outputSpacing[i] );
  }
#ifndef NDEBUG
  std::cout << "output params obtained." << std::endl;
#endif
  GPUInputImageType::Pointer itkGPUImage = GPUInputImageType::New();
  try
  {
    itkGPUImage->GraftITKImage( itkImage );
    itkGPUImage->AllocateGPU();
    itkGPUImage->GetGPUDataManager()->SetCPUBufferLock( true );
    itkGPUImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
    itkGPUImage->GetGPUDataManager()->UpdateGPUBuffer();
  }
  catch( itk::ExceptionObject & e )
  {
    std::ostringstream o_string;
    o_string  << "ERROR: Exception during creating GPU input image: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    itk::ReleaseContext();
  }

  // Create GPU copy for interpolator here
  InterpolateCopierType::GPUExplicitInterpolatorPointer gpuInterpolator;
  InterpolateCopierType::Pointer interpolateCopier = InterpolateCopierType::New();
  interpolateCopier->SetInputInterpolator( cpuInterpolator );
  interpolateCopier->Update();
  gpuInterpolator = interpolateCopier->GetModifiableExplicitOutput();
#ifndef NDEBUG 
  std::cout << "interpolator obtained." << std::endl;
#endif
  // GPU trasmform  
  TransformType::Pointer    gpuTransform;
  TransformCopierType::Pointer transformCopier = TransformCopierType::New();
  transformCopier->SetInputTransform( cpuTransform );
  transformCopier->Update();
  gpuTransform = transformCopier->GetModifiableOutput();
#ifndef NDEBUG
  std::cout << "transform obtained." << std::endl;
#endif 
  // Construct the filter
  // Use a try/catch, because construction of this filter will trigger
  // OpenCL compilation, which may fail.
  FilterType::Pointer       gpuFilter;
  try
  {
    gpuFilter = FilterType::New();
#ifndef NDEBUG
    std::cout << "obtaining GPU Filter Done" << std::endl;
#endif
  }
  catch( itk::ExceptionObject & e )
  {
    std::ostringstream o_string;
    o_string  << "Caught ITK exception during gpuFilter::New(): " << e << std::endl;
    SetLastError(o_string.str().c_str());
    itk::ReleaseContext();
    return NULL;
  }
#ifndef NDEBUG  
  std::cout << "filter obtained." << std::endl;
#endif
  try
  {
    gpuFilter->SetDefaultPixelValue( -2048 );
    gpuFilter->SetOutputSpacing( outputSpacing );
    gpuFilter->SetOutputOrigin( outputOrigin );
    gpuFilter->SetOutputDirection( outputDirection );
    gpuFilter->SetSize( outputSize );
    gpuFilter->SetOutputStartIndex( inputRegion.GetIndex() );
    
    gpuFilter->SetInput( itkGPUImage );
    gpuFilter->SetTransform(gpuTransform);
    gpuFilter->SetInterpolator(gpuInterpolator);
#ifndef NDEBUG
    std::cout << "filter settled." << std::endl;
#endif
    gpuFilter->Update();
#ifndef NDEBUG
    std::cout << "filter updated." << std::endl;
#endif  
  }
  catch( itk::ExceptionObject & e )
  {
    std::ostringstream o_string;
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    
    itk::ReleaseContext();
    return NULL;
  }

  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
CLGPUInterface::CLGPUInterface()
{
  // Create and check OpenCL context
  if( !itk::CreateContext() )
  {
    return;
  }

  // Check for the device 'double' support
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if( !context->GetDefaultDevice().HasDouble() )
  {
    std::ostringstream o_string;
    o_string << "Your OpenCL device: " << context->GetDefaultDevice().GetName()
              << ", does not support 'double' computations. Consider updating it." << std::endl;
    SetLastError(o_string.str().c_str());

    itk::ReleaseContext();
    return;
  }
}

//--------------------------------------------------------
CLGPUInterface::~CLGPUInterface()
{
#ifndef NDEBUG
    std::cout << "releasing context" << std::endl;
#endif
    itk::ReleaseContext();
}

//--------------------------------------------------------
void CLGPUInterface::SetLastError(const char* err)
{
  memset(_last_error, 0, sizeof(_last_error));     
  strcpy(_last_error, err);
}

//--------------------------------------------------------
const char* CLGPUInterface::GetLastError()
{
  return _last_error;
}


}






