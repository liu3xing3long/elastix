
// Other include files
#include <iomanip> // setprecision, etc.
#include <sstream>

#include "elxCLGPUInterface.h"
#include "itkCLGPUInterfaceHelper.h"

// ITK include files
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkOutputWindow.h"
#include "itkTimeProbe.h"

// GPU include files
#include "itkGPUResampleImageFilter.h"
#include "itkGPUBinaryDilateImageFilter.h"
#include "itkGPUBinaryErodeImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkGPUBinaryThresholdImageFilter.h"
#include "itkGPUThresholdImageFilter.h"
#include "itkGPUMeanImageFilter.h"
#include "itkGPURecursiveGaussianImageFilter.h"
#include "itkGPUDiscreteGaussianImageFilter.h"
#include "itkGPUGradientAnisotropicDiffusionImageFilter.h"

namespace elastix
{

//--------------------------------------------------------
template<typename TInput, typename TOutput> typename TOutput::Pointer 
CraftGPUImage(typename TInput::Pointer itkImage)
{
  std::ostringstream o_string;
  
  typename TOutput::Pointer itkGPUImage = TOutput::New();
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
    o_string << "ERROR: Exception during creating GPU input image: " << e << std::endl;
    // SetLastError(o_string.str().c_str());
    itk::ReleaseContext();
    return NULL;
  }

  return itkGPUImage;
}

//------------------------------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::GPUMemoryTest(CPUInputImageType::Pointer itkImage)
{
#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif

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
GPUOutputImageType::Pointer 
CLGPUInterface::Resample(CPUInputImageType::Pointer itkImage, std::vector<float> outSpacing)
{
  std::ostringstream o_string;
  
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

#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif

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
    tmp1 = outSpacing[i] / inputSpacing[i];
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
  // GPUInputImageType::Pointer itkGPUImage = GPUInputImageType::New();
  // try
  // {
  //   itkGPUImage->GraftITKImage( itkImage );
  //   itkGPUImage->AllocateGPU();
  //   itkGPUImage->GetGPUDataManager()->SetCPUBufferLock( true );
  //   itkGPUImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
  //   itkGPUImage->GetGPUDataManager()->UpdateGPUBuffer();
  // }
  // catch( itk::ExceptionObject & e )
  // {
  //   o_string  << "ERROR: Exception during creating GPU input image: " << e << std::endl;
  //   SetLastError(o_string.str().c_str());
  //   itk::ReleaseContext();
  //   return NULL;
  // }
  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

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
    gpuFilter->SetTransform( gpuTransform );
    gpuFilter->SetInterpolator( gpuInterpolator );
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
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    
    itk::ReleaseContext();
    return NULL;
  }

  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::Threshold(CPUInputImageType::Pointer itkImage, double lowerThreshold, double upperThreshold, double outsideValue)
{
  // some constants used and not directly assigned outside the function are placed here
  std::ostringstream o_string;

#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif
  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

  typedef itk::GPUThresholdImageFilter<GPUInputImageType> GPUFilterType;
  GPUFilterType::Pointer gpuFilter = GPUFilterType::New();

  try
  {
    // gpuFilter->SetUpper( upperThreshold );
    // gpuFilter->SetLower( lowerThreshold );
    gpuFilter->SetOutsideValue( outsideValue );
    gpuFilter->ThresholdOutside(lowerThreshold, upperThreshold);
    gpuFilter->SetInput( itkGPUImage );

#ifndef NDEBUG
    std::cout << "filter settled." << std::endl;
#endif
    gpuFilter->Update();
    gpuFilter->GetOutput()->UpdateBuffers(); // synchronization point
#ifndef NDEBUG
    std::cout << "filter updated." << std::endl;
#endif  
  }
  catch( itk::ExceptionObject & e )
  {
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    
    itk::ReleaseContext();
    return NULL;
  }
  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
BinGPUOutputImageType::Pointer 
CLGPUInterface::BinaryDilate(BinCPUInputImageType::Pointer itkImage, int iRadius)
{
  // some constants used and not directly assigned outside the function are placed here
  int iForegroundVal = 1, iBackgroundVal = 0, iEnableBoundary = 0;
  std::ostringstream o_string;

#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif

  // // construct gpu image
  // BinGPUInputImageType::Pointer itkGPUImage = BinGPUInputImageType::New();
  // try
  // {
  //   itkGPUImage->GraftITKImage( itkImage );
  //   itkGPUImage->AllocateGPU();
  //   itkGPUImage->GetGPUDataManager()->SetCPUBufferLock( true );
  //   itkGPUImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
  //   itkGPUImage->GetGPUDataManager()->UpdateGPUBuffer();
  // }
  // catch( itk::ExceptionObject & e )
  // {
  //   o_string  << "ERROR: Exception during creating GPU input image: " << e << std::endl;
  //   SetLastError(o_string.str().c_str());
  //   itk::ReleaseContext();
  //   return NULL;
  // }

  BinGPUInputImageType::Pointer itkGPUImage = CraftGPUImage<BinCPUInputImageType, BinGPUInputImageType> (itkImage);

  // construct filter
  typedef itk::BinaryBallStructuringElement< InterBinGPUPixType, InterGPUDIM > SRType;
  SRType kernel;
  kernel.SetRadius( iRadius );
  kernel.CreateStructuringElement();

  typedef itk::GPUBinaryDilateImageFilter< BinGPUInputImageType, BinGPUOutputImageType, SRType > FilterType;
  FilterType::Pointer gpuFilter = FilterType::New();
  gpuFilter->SetInput( itkGPUImage );
  gpuFilter->SetKernel( kernel );

  // test default values
  if ( gpuFilter->GetBackgroundValue( ) != itk::NumericTraits< InterBinGPUPixType >::NonpositiveMin() )
  {
    o_string << "Wrong default background value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }
  if ( gpuFilter->GetForegroundValue( ) != itk::NumericTraits< InterBinGPUPixType >::max() )
  {
    o_string << "Wrong default foreground value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }
  if ( gpuFilter->GetDilateValue( ) != itk::NumericTraits< InterBinGPUPixType >::max() )
  {
    o_string << "Wrong default dilate value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }
  if ( gpuFilter->GetBoundaryToForeground( ) != false )
  {
    o_string << "Wrong default BoundaryToForeground value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  //Exercise Set/Get methods for Background Value
  gpuFilter->SetForegroundValue( iForegroundVal );
  if ( gpuFilter->GetForegroundValue( ) != iForegroundVal )
  {
    o_string << "Set/Get Foreground value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  // the same with the alias
  gpuFilter->SetDilateValue( iForegroundVal );
  if ( gpuFilter->GetDilateValue( ) != iForegroundVal )
  {
    o_string << "Set/Get Dilate value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  gpuFilter->SetBackgroundValue( iBackgroundVal );
  if ( gpuFilter->GetBackgroundValue( ) != iBackgroundVal )
  {
    o_string << "Set/Get Background value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  gpuFilter->SetBoundaryToForeground( iEnableBoundary );
  if ( gpuFilter->GetBoundaryToForeground( ) != (bool)(iEnableBoundary) )
  {
    o_string << "Set/Get BoundaryToForeground value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

 // finally, generating output
  try
  {
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
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    
    itk::ReleaseContext();
    return NULL;
  }

  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
BinGPUOutputImageType::Pointer 
CLGPUInterface::BinaryErode(BinCPUInputImageType::Pointer itkImage, int iRadius)
{
  // some constants used and not directly assigned outside the function are placed here
  // NOTE, we enable the iEnableBoundary here to make sure the eroded image not get smaller
  // which is like the 'padding' in deep learning  
  int iForegroundVal = 1, iBackgroundVal = 0, iEnableBoundary = 1;
  std::ostringstream o_string;

#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif
  
  // // construct gpu image
  // BinGPUInputImageType::Pointer itkGPUImage = BinGPUInputImageType::New();
  // try
  // {
  //   itkGPUImage->GraftITKImage( itkImage );
  //   itkGPUImage->AllocateGPU();
  //   itkGPUImage->GetGPUDataManager()->SetCPUBufferLock( true );
  //   itkGPUImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
  //   itkGPUImage->GetGPUDataManager()->UpdateGPUBuffer();
  // }
  // catch( itk::ExceptionObject & e )
  // {
  //   o_string  << "ERROR: Exception during creating GPU input image: " << e << std::endl;
  //   SetLastError(o_string.str().c_str());
  //   itk::ReleaseContext();
  //   return NULL;
  // }
  BinGPUInputImageType::Pointer itkGPUImage = CraftGPUImage<BinCPUInputImageType, BinGPUInputImageType> (itkImage);

  // construct filter
  typedef itk::BinaryBallStructuringElement< InterBinGPUPixType, InterGPUDIM > SRType;
  SRType kernel;
  kernel.SetRadius( iRadius );
  kernel.CreateStructuringElement();

  typedef itk::GPUBinaryErodeImageFilter< BinGPUInputImageType, BinGPUOutputImageType, SRType > FilterType;
  FilterType::Pointer gpuFilter = FilterType::New();
  gpuFilter->SetInput( itkGPUImage );
  gpuFilter->SetKernel( kernel );

  // test default values
  if ( gpuFilter->GetBackgroundValue() != itk::NumericTraits< InterBinGPUPixType >::NonpositiveMin() )
  {
    o_string << "Wrong default background value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }
  if ( gpuFilter->GetForegroundValue() != itk::NumericTraits< InterBinGPUPixType >::max() )
  {
    o_string << "Wrong default foreground value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }
  if ( gpuFilter->GetErodeValue() != itk::NumericTraits< InterBinGPUPixType >::max() )
  {
    o_string << "Wrong default erode value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }
  if ( gpuFilter->GetBoundaryToForeground() != true )
  {
    o_string << "Wrong default BoundaryToForeground value." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  //Exercise Set/Get methods for Background Value
  gpuFilter->SetForegroundValue( iForegroundVal );
  if ( gpuFilter->GetForegroundValue() != iForegroundVal )
  {
    o_string << "Set/Get Foreground value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  // the same with the alias
  gpuFilter->SetErodeValue( iForegroundVal );
  if ( gpuFilter->GetErodeValue() != iForegroundVal )
  {
    o_string << "Set/Get Erode value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  gpuFilter->SetBackgroundValue( iBackgroundVal );
  if ( gpuFilter->GetBackgroundValue() != iBackgroundVal )
  {
    o_string << "Set/Get Background value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  gpuFilter->SetBoundaryToForeground( iEnableBoundary );
  if ( gpuFilter->GetBoundaryToForeground() != (bool)( iEnableBoundary ) )
  {
    o_string << "Set/Get BoundaryToForeground value problem." << std::endl;
    SetLastError(o_string.str().c_str());
    return NULL;
  }

  // finally, generating output
  try
  {
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
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    
    itk::ReleaseContext();
    return NULL;
  }

  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
BinGPUOutputImageType::Pointer 
CLGPUInterface::BinaryThreshold(CPUInputImageType::Pointer itkImage, 
                                double lowerThreshold /* = 0.0 */, double upperThreshold /* = 255.0 */, 
                                uint8_t insideValue /* = 1u */, uint8_t outsideValue /* = 0u */)
{
  // some constants used and not directly assigned outside the function are placed here
  std::ostringstream o_string;

#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif
  // // construct gpu image
  // GPUInputImageType::Pointer itkGPUImage = GPUInputImageType::New();
  // try
  // {
  //   itkGPUImage->GraftITKImage( itkImage );
  //   itkGPUImage->AllocateGPU();
  //   itkGPUImage->GetGPUDataManager()->SetCPUBufferLock( true );
  //   itkGPUImage->GetGPUDataManager()->SetGPUDirtyFlag( true );
  //   itkGPUImage->GetGPUDataManager()->UpdateGPUBuffer();
  // }
  // catch( itk::ExceptionObject & e )
  // {
  //   o_string  << "ERROR: Exception during creating GPU input image: " << e << std::endl;
  //   SetLastError(o_string.str().c_str());
  //   itk::ReleaseContext();
  //   return NULL;
  // }
  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

  typedef itk::GPUBinaryThresholdImageFilter<GPUInputImageType, BinGPUOutputImageType> GPUFilterType;
  GPUFilterType::Pointer gpuFilter = GPUFilterType::New();

  try
  {
    gpuFilter->SetOutsideValue( outsideValue );
    gpuFilter->SetInsideValue( insideValue );
    gpuFilter->SetUpperThreshold( upperThreshold );
    gpuFilter->SetLowerThreshold( lowerThreshold );
    gpuFilter->SetInput( itkGPUImage );

#ifndef NDEBUG
    std::cout << "filter settled." << std::endl;
#endif
    gpuFilter->Update();
    gpuFilter->GetOutput()->UpdateBuffers(); // synchronization point
#ifndef NDEBUG
    std::cout << "filter updated." << std::endl;
#endif  
  }
  catch( itk::ExceptionObject & e )
  {
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    
    itk::ReleaseContext();
    return NULL;
  }
  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::Median(CPUInputImageType::Pointer itkImage, const std::vector<unsigned int>& radius)
{
#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif

  std::ostringstream o_string;

  o_string << "function not implemented..." << std::endl;
  SetLastError(o_string.str().c_str());
  return NULL;
}

//--------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::Mean(CPUInputImageType::Pointer itkImage, const std::vector<unsigned int>& radius)
{
#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif
  // some constants used and not directly assigned outside the function are placed here
  std::ostringstream o_string;
  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

  typedef itk::GPUMeanImageFilter< GPUInputImageType, GPUOutputImageType > GPUFilterType;
  GPUFilterType::Pointer gpuFilter = GPUFilterType::New();

  // setting down radiuses, should be transformed to itk size format
  GPUFilterType::SizeType szRadius;
  for (unsigned int idx = 0; idx < radius.size(); idx++){
    szRadius[idx] = radius[idx];
  }

  try
  {
    gpuFilter->SetRadius( szRadius );
    gpuFilter->SetInput( itkGPUImage );

#ifndef NDEBUG
    std::cout << "filter settled." << std::endl;
#endif
    gpuFilter->Update();
    gpuFilter->GetOutput()->UpdateBuffers(); // synchronization point
#ifndef NDEBUG
    std::cout << "filter updated." << std::endl;
#endif  
  }
  catch( itk::ExceptionObject & e )
  {
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    itk::ReleaseContext();
    return NULL;
  }
  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::RecursiveGaussian(CPUInputImageType::Pointer itkImage, double sigma, bool normalizeAcrossScale, 
                                 unsigned int order, unsigned int direction)
{
#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif
  // some constants used and not directly assigned outside the function are placed here
  std::ostringstream o_string;
  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

  typedef itk::GPURecursiveGaussianImageFilter< GPUInputImageType, GPUOutputImageType > GPUFilterType;
  GPUFilterType::Pointer gpuFilter = GPUFilterType::New();
  GPUFilterType::OrderEnumType gorder;
  switch (order)
  {
    case 0:
      gorder = GPUFilterType::ZeroOrder;
      break;
    case 1:
      gorder = GPUFilterType::FirstOrder;
      break;
    case 2:
      gorder = GPUFilterType::SecondOrder;
      break;
    default:
      o_string << "giving unsupported order type, giving order " << order << " should be in range [0, 2]" << std::endl;
      SetLastError(o_string.str().c_str());
      return NULL;
  }

  try
  {
    gpuFilter->SetOrder( gorder );
    gpuFilter->SetNormalizeAcrossScale( normalizeAcrossScale );
    gpuFilter->SetDirection( direction );
    gpuFilter->SetSigma( sigma );
    gpuFilter->SetInput( itkGPUImage );
#ifndef NDEBUG
    std::cout << "filter settled." << std::endl;
#endif
    gpuFilter->Update();
    gpuFilter->GetOutput()->UpdateBuffers(); // synchronization point
#ifndef NDEBUG
    std::cout << "filter updated." << std::endl;
#endif  
  }
  catch( itk::ExceptionObject & e )
  {
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    itk::ReleaseContext();
    return NULL;
  }
  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::DiscreteGaussian(CPUInputImageType::Pointer itkImage, double variance,
                                unsigned int maximumKernelWidth, double maximumError, bool useImageSpacing)
{
#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif

  // some constants used and not directly assigned outside the function are placed here
  std::ostringstream o_string;

  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

  typedef itk::GPUDiscreteGaussianImageFilter< GPUInputImageType, GPUOutputImageType > GPUFilterType;
  GPUFilterType::Pointer gpuFilter = GPUFilterType::New();

  try
  {
    gpuFilter->SetVariance( variance );
    gpuFilter->SetMaximumError( maximumError );
    gpuFilter->SetMaximumKernelWidth( maximumKernelWidth );
    gpuFilter->SetUseImageSpacing( useImageSpacing );

    // working towards the whole dim
    gpuFilter->SetFilterDimensionality( GPUInputImageType::ImageDimension );

#ifndef NDEBUG
    std::cout << "filter settled." << std::endl;
#endif
    gpuFilter->Update();
    gpuFilter->GetOutput()->UpdateBuffers(); // synchronization point
#ifndef NDEBUG
    std::cout << "filter updated." << std::endl;
#endif  
  }
  catch( itk::ExceptionObject & e )
  {
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    itk::ReleaseContext();
    return NULL;
  }
  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::DiscreteGaussian(CPUInputImageType::Pointer itkImage, const std::vector< double > &variance,
                                unsigned int maximumKernelWidth, const std::vector< double > &maximumError,
                                bool useImageSpacing)
{
#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif
  std::ostringstream o_string;

  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

  typedef itk::GPUDiscreteGaussianImageFilter< GPUInputImageType, GPUOutputImageType > GPUFilterType;
  GPUFilterType::Pointer gpuFilter = GPUFilterType::New();

  // setting down radiuses, should be transformed to itk size format
  GPUFilterType::ArrayType itkVar;
  for (unsigned int idx = 0; idx < variance.size(); idx++){
    itkVar[idx] = variance[idx];
  }

  GPUFilterType::ArrayType itkError;
  for (unsigned int idx = 0; idx < maximumError.size(); idx++){
    itkError[idx] = maximumError[idx];
  }

  try
  {
    gpuFilter->SetVariance( itkVar );
    gpuFilter->SetMaximumError( itkError );
    gpuFilter->SetMaximumKernelWidth( maximumKernelWidth );
    gpuFilter->SetUseImageSpacing( useImageSpacing );

    // working towards the whole dim
    gpuFilter->SetFilterDimensionality( GPUInputImageType::ImageDimension );

#ifndef NDEBUG
    std::cout << "filter settled." << std::endl;
#endif
    gpuFilter->Update();
    gpuFilter->GetOutput()->UpdateBuffers(); // synchronization point
#ifndef NDEBUG
    std::cout << "filter updated." << std::endl;
#endif  
  }
  catch( itk::ExceptionObject & e )
  {
    o_string << "ERROR: " << e << std::endl;
    SetLastError(o_string.str().c_str());
    itk::ReleaseContext();
    return NULL;
  }
  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
GPUOutputImageType::Pointer 
CLGPUInterface::GradientAnisotropicDiffusion(CPUInputImageType::Pointer itkImage, double timeStep,
                                            double conductanceParameter, unsigned int conductanceScalingUpdateInterval,
                                            uint32_t numberOfIterations)
{
#ifndef NDEBUG
  std::cout << "entering function " << __FUNCTION__ << std::endl;
#endif
  std::ostringstream o_string;

  GPUInputImageType::Pointer itkGPUImage = CraftGPUImage<CPUInputImageType, GPUInputImageType> (itkImage);

typedef itk::GPUGradientAnisotropicDiffusionImageFilter< GPUInputImageType, GPUOutputImageType > GPUFilterType;
GPUFilterType::Pointer gpuFilter = GPUFilterType::New();

  try
  {
    gpuFilter->SetInput( itkGPUImage );
    gpuFilter->SetNumberOfIterations( numberOfIterations );
    gpuFilter->SetTimeStep( timeStep );//125 );
    gpuFilter->SetConductanceParameter( conductanceParameter );
    gpuFilter->SetConductanceScalingUpdateInterval( conductanceScalingUpdateInterval );
    gpuFilter->UseImageSpacingOn();
#ifndef NDEBUG
  std::cout << "filter settled" << std::endl;
#endif
    gpuFilter->Update();
    gpuFilter->GetOutput()->UpdateBuffers(); // synchronization point
#ifndef NDEBUG
  std::cout << "filter updated" << std::endl;
#endif
  }
  catch (itk::ExceptionObject& excp)
  {
    o_string << "Caught exception during setting params" << excp << std::endl;
    return NULL;
  }

  return gpuFilter->GetOutput();
}

//--------------------------------------------------------
CLGPUInterface::CLGPUInterface()
{
  Init();
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
bool CLGPUInterface::IsGPUEnabled()
{
  itk::OpenCLDevice device;

  if( !device.IsNull() )
  {
    return EXIT_FAILURE;
  }

  // Get all devices
  std::list< itk::OpenCLDevice >       gpus;
  const std::list< itk::OpenCLDevice > devices = itk::OpenCLDevice::GetAllDevices();
  for( std::list< itk::OpenCLDevice >::const_iterator dev = devices.begin(); dev != devices.end(); ++dev )
  {
    if( ( ( *dev ).GetDeviceType() & itk::OpenCLDevice::GPU ) != 0 )
    {
      gpus.push_back( *dev );
#ifndef NDEBUG
      std::cout << ( *dev );
#endif
    }
  }

  if (gpus.size() > 0)
  {
    return true;
  }
  else
  {
    return false;
  }
}

//--------------------------------------------------------
bool CLGPUInterface::Init()
{
  std::ostringstream o_string;
  // Create and check OpenCL context
  if( !itk::CreateContext() )
  {
    o_string << "Creating OpenGL Context failed ! Check your GPU! " << std::endl;
    SetLastError(o_string.str().c_str());
    return false;
  }

  // Check for the device 'double' support
  itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
  if( !context->GetDefaultDevice().HasDouble() )
  {
    o_string << "Your OpenCL device: " << context->GetDefaultDevice().GetName()
              << ", does not support 'double' computations. Consider updating it." << std::endl;
    SetLastError(o_string.str().c_str());

    itk::ReleaseContext();
    return false;
  }
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






