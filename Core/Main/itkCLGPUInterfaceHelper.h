/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
//
// \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
// Department of Radiology, Leiden, The Netherlands
//
// \note This work was funded by the Netherlands Organisation for
// Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
//
#ifndef __itkCLGPUInterfaceHelper_h
#define __itkCLGPUInterfaceHelper_h

#include <string>
#include <vector>
#include <iomanip>
#include <time.h>
#include <fstream>
#include <itksys/SystemTools.hxx>

#if defined( _WIN32 )
#include <io.h>
#endif

// ITK includes
#include "itkImage.h"
#include "itkTimeProbe.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageToImageFilter.h"

// OpenCL includes
#include "itkOpenCLContext.h"
#include "itkOpenCLDevice.h"
#include "itkOpenCLLogger.h"
#include "itkOpenCLKernels.h"
#include "itkOpenCLPlatform.h"

// GPU copiers
#include "itkGPUTransformCopier.h"
#include "itkGPUAdvancedCombinationTransformCopier.h"
#include "itkGPUInterpolatorCopier.h"

// GPU factory includes
#include "itkGPUImageFactory.h"
#include "itkGPUResampleImageFilterFactory.h"
#include "itkGPUCastImageFilterFactory.h"
#include "itkGPUAffineTransformFactory.h"
#include "itkGPUTranslationTransformFactory.h"
#include "itkGPUBSplineTransformFactory.h"
#include "itkGPUEuler3DTransformFactory.h"
#include "itkGPUSimilarity3DTransformFactory.h"

// 
#include "itkGPUNearestNeighborInterpolateImageFunctionFactory.h"
#include "itkGPULinearInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineInterpolateImageFunctionFactory.h"
#include "itkGPUBSplineDecompositionImageFilterFactory.h"

// elastix GPU factory includes
#include "itkGPUAdvancedCombinationTransformFactory.h"
#include "itkGPUAdvancedMatrixOffsetTransformBaseFactory.h"
#include "itkGPUAdvancedTranslationTransformFactory.h"
#include "itkGPUAdvancedBSplineDeformableTransformFactory.h"
#include "itkGPUAdvancedSimilarity3DTransformFactory.h"
#include "itkGPUAdvancedEuler3DTransformFactory.h"


//------------------------------------------------------------------------------
// Definition of the OCLImageDims
struct OCLImageDims
{
    itkStaticConstMacro( Support1D, bool, false );
    itkStaticConstMacro( Support2D, bool, false );
    itkStaticConstMacro( Support3D, bool, true );
};

#define ITK_OPENCL_COMPARE( actual, expected )                                    \
  if( !itk::Compare( actual, expected, #actual, #expected, __FILE__, __LINE__ ) ) \
    itkGenericExceptionMacro( << "Compared values are not the same" )             \


namespace itk
{
//------------------------------------------------------------------------------
template< typename T >
inline bool
Compare( T const &t1, T const &t2, const char *actual, const char *expected, const char *file, int line )
{
    return (t1 == t2) ? true : false;
}

//------------------------------------------------------------------------------
bool
CreateContext( const std::vector< unsigned int > &device_ids, OpenCLDevice::DeviceType deviceType = OpenCLDevice::GPU,
               OpenCLPlatform::VendorType vendorType = OpenCLPlatform::NVidia )
{
    // Create and check OpenCL context
    OpenCLContext::Pointer context = OpenCLContext::GetInstance();
    std::list< OpenCLDevice > devices = OpenCLDevice::GetDevices( deviceType, vendorType );
    unsigned int devices_count = devices.size();
    
    std::list< OpenCLDevice > selected_devices;
    for ( unsigned int d_idx = 0; d_idx < devices_count; d_idx++ )
    {
        unsigned int device_id = device_ids[d_idx];
        if ( device_id >= devices_count )
        {
            std::cerr << "selecting device id " << device_id << " larger than devices count " << devices_count
                      << " skipping..." << std::endl;
            
        }
        else
        {
            std::list< OpenCLDevice >::iterator iter  = devices.begin();
            std::advance( iter, device_id );
            selected_devices.push_back( *iter );
        }
    }
    
    context->Create( selected_devices );
    
    if ( !context->IsCreated() )
    {
        std::cerr << "OpenCL-enabled device is not present." << std::endl;
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
bool
CreateContext()
{
    // Create and check OpenCL context
    OpenCLContext::Pointer context = OpenCLContext::GetInstance();

#if defined( OPENCL_USE_INTEL_CPU ) || defined( OPENCL_USE_AMD_CPU )
    context->Create( OpenCLContext::DevelopmentSingleMaximumFlopsDevice );
#else
    context->Create( OpenCLContext::SingleMaximumFlopsDevice );
#endif
    
    if ( !context->IsCreated() )
    {
        std::cerr << "OpenCL-enabled device is not present." << std::endl;
        return false;
    }
    
    return true;
}

//------------------------------------------------------------------------------
void
ReleaseContext()
{
    itk::OpenCLContext::Pointer context = itk::OpenCLContext::GetInstance();
    if ( context->IsCreated() )
    {
        context->Release();
    }
}

//------------------------------------------------------------------------------
void
CreateOpenCLLogger( const std::string &prefixFileName )
{
    /** Create the OpenCL logger */
    OpenCLLogger::Pointer logger = OpenCLLogger::GetInstance();
    logger->SetLogFileNamePrefix( prefixFileName );
    logger->SetOutputDirectory( OpenCLKernelsDebugDirectory );
}


// //------------------------------------------------------------------------------
// void
// SetupForDebugging()
// {
//   TestOutputWindow::Pointer tow = TestOutputWindow::New();
//   OutputWindow::SetInstance( tow );

// #if ( defined( _WIN32 ) && defined( _DEBUG ) ) || !defined( NDEBUG )
//   Object::SetGlobalWarningDisplay( true );
//   std::cout << "INFO: test called Object::SetGlobalWarningDisplay(true)\n";
// #endif
// }


//------------------------------------------------------------------------------
void
ITKObjectEnableWarnings( Object *object )
{
#if (defined( _WIN32 ) && defined( _DEBUG )) || !defined( NDEBUG )
    object->SetDebug( true );
    std::cout << "INFO: " << object->GetNameOfClass() << " called SetDebug(true);\n";
#endif
}

//------------------------------------------------------------------------------
// Get current date, format is m-d-y
const std::string
GetCurrentDate()
{
    time_t now = time( 0 );
    struct tm tstruct;
    char buf[80];

#if !defined( _WIN32 ) || defined( __CYGWIN__ )
    tstruct = *localtime( &now );
#else
    localtime_s( &tstruct, &now );
#endif
    
    // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
    // for more information about date/time format
    strftime( buf, sizeof( buf ), "%m-%d-%y", &tstruct );
    
    return buf;
}

//------------------------------------------------------------------------------
// Get the name of the log file
const std::string
GetLogFileName()
{
    OpenCLContext::Pointer context = OpenCLContext::GetInstance();
    std::string fileName = "CPUGPULog-" + itk::GetCurrentDate() + "-";
    std::string deviceName = context->GetDefaultDevice().GetName();
    std::replace( deviceName.begin(), deviceName.end(), ' ', '-' ); // replace spaces
    deviceName.erase( deviceName.end() - 1, deviceName.end() );     // remove end of line
    
    switch ( context->GetDefaultDevice().GetDeviceType() )
    {
        case OpenCLDevice::Default:
            fileName.append( "Default" );
            break;
        case OpenCLDevice::CPU:
            fileName.append( "CPU" );
            break;
        case OpenCLDevice::GPU:
            fileName.append( "GPU" );
            break;
        case OpenCLDevice::Accelerator:
            fileName.append( "Accelerator" );
            break;
        case OpenCLDevice::All:
            fileName.append( "All" );
            break;
        default:
            fileName.append( "Unknown" );
            break;
    }
    
    fileName.append( "-" );
    fileName.append( deviceName );
    fileName.append( ".txt" );
    
    return fileName;
}

//------------------------------------------------------------------------------
// Helper function to compute RMSE
template< class TScalarType, class CPUImageType, class GPUImageType >
TScalarType
ComputeRMSE( const CPUImageType *cpuImage, const GPUImageType *gpuImage, TScalarType &rmsRelative )
{
    ImageRegionConstIterator< CPUImageType > cit( cpuImage, cpuImage->GetLargestPossibleRegion() );
    ImageRegionConstIterator< GPUImageType > git( gpuImage, gpuImage->GetLargestPossibleRegion() );
    
    TScalarType rmse = 0.0;
    TScalarType sumCPUSquared = 0.0;
    
    for ( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
    {
        TScalarType cpu = static_cast< TScalarType >( cit.Get() );
        TScalarType err = cpu - static_cast< TScalarType >( git.Get() );
        rmse += err * err;
        sumCPUSquared += cpu * cpu;
    }
    
    rmse = vcl_sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
    rmsRelative = rmse / vcl_sqrt( sumCPUSquared / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
    
    return rmse;
} // end ComputeRMSE()


//------------------------------------------------------------------------------
// Helper function to compute RMSE
template< class TScalarType, class CPUImageType, class GPUImageType >
TScalarType
ComputeRMSE2( const CPUImageType *cpuImage, const GPUImageType *gpuImage, const float &threshold )
{
    ImageRegionConstIterator< CPUImageType > cit( cpuImage, cpuImage->GetLargestPossibleRegion() );
    ImageRegionConstIterator< GPUImageType > git( gpuImage, gpuImage->GetLargestPossibleRegion() );
    
    TScalarType rmse = 0.0;
    
    for ( cit.GoToBegin(), git.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git )
    {
        TScalarType err = static_cast< TScalarType >( cit.Get() ) - static_cast< TScalarType >( git.Get() );
        if ( err > threshold )
        {
            rmse += err * err;
        }
    }
    rmse = vcl_sqrt( rmse / cpuImage->GetLargestPossibleRegion().GetNumberOfPixels() );
    return rmse;
} // end ComputeRMSE2()


//------------------------------------------------------------------------------
// Helper function to get test result from output images
template< class TScalarType, class CPUImageType, class GPUImageType >
void
GetTestOutputResult( const CPUImageType *cpuImage, const GPUImageType *gpuImage, const float allowedRMSerror,
                     TScalarType &rmsError, TScalarType &rmsRelative, bool &testPassed, const bool skipCPU,
                     const bool skipGPU, const TimeProbe::TimeStampType cpuTime, const TimeProbe::TimeStampType gpuTime,
                     const bool updateExceptionCPU, const bool updateExceptionGPU )
{
    rmsError = 0.0;
    rmsRelative = 0.0;
    testPassed = true;
    if ( updateExceptionCPU || updateExceptionGPU )
    {
        testPassed = false;
    }
    else
    {
        if ( !skipCPU && !skipGPU && cpuImage && gpuImage )
        {
            rmsError = ComputeRMSE< TScalarType, CPUImageType, GPUImageType >( cpuImage, gpuImage, rmsRelative );
            
            std::cout << ", speed up " << (cpuTime / gpuTime) << std::endl;
            std::cout << std::fixed << std::setprecision( 8 );
            std::cout << "Maximum allowed RMS Error: " << allowedRMSerror << std::endl;
            std::cout << "Computed real   RMS Error: " << rmsError << std::endl;
            std::cout << "Computed real  nRMS Error: " << rmsRelative << std::endl;
            
            testPassed = (rmsError <= allowedRMSerror);
        }
    }
}

//------------------------------------------------------------------------------
// Helper function to get test result from filters
template< class TScalarType, class ImageToImageFilterType, class OutputImage >
void
GetTestFilterResult( typename ImageToImageFilterType::Pointer &cpuFilter,
                     typename ImageToImageFilterType::Pointer &gpuFilter, const float allowedRMSerror,
                     TScalarType &rmsError, TScalarType &rmsRelative, bool &testPassed, const bool skipCPU,
                     const bool skipGPU, const TimeProbe::TimeStampType cpuTime, const TimeProbe::TimeStampType gpuTime,
                     const bool updateExceptionCPU, const bool updateExceptionGPU, const unsigned int outputindex = 0 )
{
    rmsError = 0.0;
    testPassed = true;
    if ( updateExceptionCPU || updateExceptionGPU )
    {
        testPassed = false;
    }
    else
    {
        if ( !skipCPU && !skipGPU && cpuFilter.IsNotNull() && gpuFilter.IsNotNull() )
        {
            if ( outputindex == 0 )
            {
                rmsError = ComputeRMSE< TScalarType, OutputImage, OutputImage >( cpuFilter->GetOutput(),
                                                                                 gpuFilter->GetOutput(), rmsRelative );
            }
            else
            {
                rmsError = ComputeRMSE< TScalarType, OutputImage, OutputImage >( cpuFilter->GetOutput( outputindex ),
                                                                                 gpuFilter->GetOutput( outputindex ),
                                                                                 rmsRelative );
            }
            
            std::cout << ", speed up " << (cpuTime / gpuTime) << std::endl;
            std::cout << std::fixed << std::setprecision( 8 );
            std::cout << "Maximum allowed RMS Error: " << allowedRMSerror << std::endl;
            std::cout << "Computed real   RMS Error: " << rmsError << std::endl;
            std::cout << "Computed real  nRMS Error: " << rmsRelative << std::endl;
            
            testPassed = (rmsError <= allowedRMSerror);
        }
    }
}

//------------------------------------------------------------------------------
// Helper function to compute RMSE with masks
template< class TScalarType, class CPUImageType, class GPUImageType, class MaskImageType >
TScalarType
ComputeRMSE( const CPUImageType *cpuImage, const GPUImageType *gpuImage, const MaskImageType *cpuImageMask,
             const MaskImageType *gpuImageMask, TScalarType &rmsRelative )
{
    ImageRegionConstIterator< CPUImageType > cit( cpuImage, cpuImage->GetLargestPossibleRegion() );
    ImageRegionConstIterator< GPUImageType > git( gpuImage, gpuImage->GetLargestPossibleRegion() );
    
    ImageRegionConstIterator< MaskImageType > mcit( cpuImageMask, cpuImageMask->GetLargestPossibleRegion() );
    ImageRegionConstIterator< MaskImageType > mgit( gpuImageMask, gpuImageMask->GetLargestPossibleRegion() );
    
    TScalarType rmse = 0.0;
    TScalarType sumCPUSquared = 0.0;
    std::size_t count = 0;
    for ( cit.GoToBegin(), git.GoToBegin(), mcit.GoToBegin(), mgit.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git, ++mcit, ++mgit )
    {
        if ( (mcit.Get() == NumericTraits< typename MaskImageType::PixelType >::OneValue()) &&
             (mgit.Get() == NumericTraits< typename MaskImageType::PixelType >::OneValue()) )
        {
            TScalarType cpu = static_cast< TScalarType >( cit.Get() );
            TScalarType err = cpu - static_cast< TScalarType >( git.Get() );
            rmse += err * err;
            sumCPUSquared += cpu * cpu;
            count++;
        }
    }
    
    if ( count == 0 )
    {
        rmsRelative = 0.0;
        return 0.0;
    }
    
    rmse = vcl_sqrt( rmse / count );
    rmsRelative = rmse / vcl_sqrt( sumCPUSquared / count );
    return rmse;
} // end ComputeRMSE()


//------------------------------------------------------------------------------
// Helper function to compute RMSE with masks and threshold
template< class TScalarType, class CPUImageType, class GPUImageType, class MaskImageType >
TScalarType
ComputeRMSE2( const CPUImageType *cpuImage, const GPUImageType *gpuImage, const MaskImageType *cpuImageMask,
              const MaskImageType *gpuImageMask, const float threshold, TScalarType &rmsRelative )
{
    ImageRegionConstIterator< CPUImageType > cit( cpuImage, cpuImage->GetLargestPossibleRegion() );
    ImageRegionConstIterator< GPUImageType > git( gpuImage, gpuImage->GetLargestPossibleRegion() );
    
    ImageRegionConstIterator< MaskImageType > mcit( cpuImageMask, cpuImageMask->GetLargestPossibleRegion() );
    ImageRegionConstIterator< MaskImageType > mgit( gpuImageMask, gpuImageMask->GetLargestPossibleRegion() );
    
    TScalarType rmse = 0.0;
    TScalarType sumCPUSquared = 0.0;
    std::size_t count = 0;
    for ( cit.GoToBegin(), git.GoToBegin(), mcit.GoToBegin(), mgit.GoToBegin(); !cit.IsAtEnd(); ++cit, ++git, ++mcit, ++mgit )
    {
        if ( mcit.Get() == NumericTraits< typename MaskImageType::PixelType >::One &&
             mgit.Get() == NumericTraits< typename MaskImageType::PixelType >::OneValue() )
        {
            ++count;
            TScalarType cpu = static_cast< TScalarType >( cit.Get() );
            TScalarType err = cpu - static_cast< TScalarType >( git.Get() );
            if ( vnl_math_abs( err ) > threshold )
            {
                rmse += err * err;
            }
            sumCPUSquared += cpu * cpu;
        }
    }
    
    if ( count == 0 )
    {
        rmsRelative = 0.0;
        return 0.0;
    }
    
    rmse = vcl_sqrt( rmse / count );
    rmsRelative = rmse / vcl_sqrt( sumCPUSquared / count );
    return rmse;
} // end ComputeRMSE()


//----------------------------------------------------------------------------
// Write log file in Microsoft Excel semicolon separated format.
template< class ImageType >
void
WriteLog( const std::string &filename, const unsigned int dim, const typename ImageType::SizeType &imagesize,
          const double rmsError, const double rmsRelative, const bool testPassed, const bool exceptionGPU,
          const unsigned int numThreads, const unsigned int runTimes, const std::string &filterName,
          const TimeProbe::TimeStampType cpuTime, const TimeProbe::TimeStampType gpuTime,
          const std::string &comments = "" )
{
    const std::string s( " ; " );   // separator
    std::ofstream fout;
    const bool fileExists = itksys::SystemTools::FileExists( filename.c_str() );
    
    fout.open( filename.c_str(), std::ios_base::app );
    // If file does not exist, then print table header
    if ( !fileExists )
    {
        fout << "Filter Name" << s << "Dimension" << s << "Image Size" << s << "CPU(" << numThreads << ") (s)" << s
             << "GPU (s)" << s << "CPU/GPU Speed Ratio" << s << "RMS Error" << s << "RMS Relative" << s << "Test Passed"
             << s << "Run Times" << s << "Comments" << std::endl;
    }
    
    fout << filterName << s << dim << s;
    for ( unsigned int i = 0; i < dim; i++ )
    {
        fout << imagesize.GetSize()[i];
        if ( i < dim - 1 )
        {
            fout << "x";
        }
    }
    
    fout << s << cpuTime << s;
    
    if ( !exceptionGPU )
    {
        fout << gpuTime << s;
    }
    else
    {
        fout << "na" << s;
    }
    
    if ( !exceptionGPU )
    {
        if ( gpuTime != 0.0 )
        {
            fout << (cpuTime / gpuTime) << s;
        }
        else
        {
            fout << "0" << s;
        }
    }
    else
    {
        fout << "na" << s;
    }
    
    fout << rmsError << s;
    fout << rmsRelative << s;
    fout << (testPassed ? "Yes" : "No") << s;
    fout << runTimes << s;
    
    if ( comments.size() > 0 )
    {
        fout << comments;
    }
    else
    {
        fout << "none";
    }
    
    fout << std::endl;
    
    fout.close();
    return;
}

//------------------------------------------------------------------------------
template< typename InputImageType >
typename InputImageType::PointType
ComputeCenterOfTheImage( const typename InputImageType::ConstPointer &image )
{
    const unsigned int Dimension = image->GetImageDimension();
    
    const typename InputImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
    const typename InputImageType::IndexType index = image->GetLargestPossibleRegion().GetIndex();
    
    typedef itk::ContinuousIndex< double, InputImageType::ImageDimension > ContinuousIndexType;
    ContinuousIndexType centerAsContInd;
    for ( std::size_t i = 0; i < Dimension; i++ )
    {
        centerAsContInd[i] = static_cast< double >( index[i] ) + static_cast< double >( size[i] - 1 ) / 2.0;
    }
    
    typename InputImageType::PointType center;
    image->TransformContinuousIndexToPhysicalPoint( centerAsContInd, center );
    return center;
}

//------------------------------------------------------------------------------
template< typename InterpolatorType >
void
DefineInterpolator( typename InterpolatorType::Pointer &interpolator, const std::string &interpolatorName,
                    const unsigned int splineOrderInterpolator )
{
    // Interpolator typedefs
    typedef typename InterpolatorType::InputImageType InputImageType;
    typedef typename InterpolatorType::CoordRepType CoordRepType;
    typedef CoordRepType CoefficientType;
    
    // Typedefs for all interpolators
    typedef itk::NearestNeighborInterpolateImageFunction< InputImageType, CoordRepType > NearestNeighborInterpolatorType;
    typedef itk::LinearInterpolateImageFunction< InputImageType, CoordRepType > LinearInterpolatorType;
    typedef itk::BSplineInterpolateImageFunction< InputImageType, CoordRepType, CoefficientType > BSplineInterpolatorType;
    
    if ( interpolatorName == "NearestNeighbor" )
    {
        typename NearestNeighborInterpolatorType::Pointer tmpInterpolator = NearestNeighborInterpolatorType::New();
        interpolator = tmpInterpolator;
    }
    else if ( interpolatorName == "Linear" )
    {
        typename LinearInterpolatorType::Pointer tmpInterpolator = LinearInterpolatorType::New();
        interpolator = tmpInterpolator;
    }
    else if ( interpolatorName == "BSpline" )
    {
        typename BSplineInterpolatorType::Pointer tmpInterpolator = BSplineInterpolatorType::New();
        tmpInterpolator->SetSplineOrder( splineOrderInterpolator );
        interpolator = tmpInterpolator;
    }
}

//------------------------------------------------------------------------------
template< typename AffineTransformType >
void
DefineAffineParameters( typename AffineTransformType::ParametersType &parameters )
{
    const unsigned int Dimension = AffineTransformType::InputSpaceDimension;
    
    // Setup parameters
    parameters.SetSize( Dimension * Dimension + Dimension );
    unsigned int par = 0;
    if ( Dimension == 2 )
    {
        const double matrix[] = {0.9, 0.1, // matrix part
                                 0.2, 1.1, // matrix part
                                 0.0, 0.0, // translation
        };
        
        for ( std::size_t i = 0; i < 6; i++ )
        {
            parameters[par++] = matrix[i];
        }
    }
    else if ( Dimension == 3 )
    {
        const double matrix[] = {1.0, -0.045, 0.02,   // matrix part
                                 0.0, 1.0, 0.0,       // matrix part
                                 -0.075, 0.09, 1.0,   // matrix part
                                 -3.02, 1.3, -0.045   // translation
        };
        
        for ( std::size_t i = 0; i < 12; i++ )
        {
            parameters[par++] = matrix[i];
        }
    }
}

//------------------------------------------------------------------------------
template< typename TranslationTransformType >
void
DefineTranslationParameters( const std::size_t transformIndex,
                             typename TranslationTransformType::ParametersType &parameters )
{
    const std::size_t Dimension = TranslationTransformType::SpaceDimension;
    
    // Setup parameters
    parameters.SetSize( Dimension );
    for ( std::size_t i = 0; i < Dimension; i++ )
    {
        parameters[i] = ( double ) i * ( double ) transformIndex + ( double ) transformIndex;
    }
}

//------------------------------------------------------------------------------
template< typename BSplineTransformType >
void
DefineBSplineParameters( const std::size_t transformIndex, typename BSplineTransformType::ParametersType &parameters,
                         const typename BSplineTransformType::Pointer &transform,
                         const std::string &parametersFileName )
{
    const unsigned int numberOfParameters = transform->GetNumberOfParameters();
    const unsigned int Dimension = BSplineTransformType::SpaceDimension;
    const unsigned int numberOfNodes = numberOfParameters / Dimension;
    
    parameters.SetSize( numberOfParameters );
    
    // Open file and read parameters
    std::ifstream infile;
    infile.open( parametersFileName.c_str() );
    
    // Skip number of elements to make unique coefficients per each transformIndex
    for ( std::size_t n = 0; n < transformIndex; n++ )
    {
        double parValue;
        infile >> parValue;
    }
    
    // Read it
    for ( std::size_t n = 0; n < numberOfNodes * Dimension; n++ )
    {
        double parValue;
        infile >> parValue;
        parameters[n] = parValue;
    }
    
    infile.close();
}

//------------------------------------------------------------------------------
template< typename EulerTransformType >
void
DefineEulerParameters( const std::size_t transformIndex, typename EulerTransformType::ParametersType &parameters )
{
    const std::size_t Dimension = EulerTransformType::InputSpaceDimension;
    
    // Setup parameters
    // 2D: angle 1, translation 2
    // 3D: 6 angle, translation 3
    parameters.SetSize( EulerTransformType::ParametersDimension );
    
    // Angle
    const double angle = ( double ) transformIndex * -0.05;
    
    std::size_t par = 0;
    if ( Dimension == 2 )
    {
        // See implementation of Rigid2DTransform::SetParameters()
        parameters[0] = angle;
        ++par;
    }
    else if ( Dimension == 3 )
    {
        // See implementation of Rigid3DTransform::SetParameters()
        for ( std::size_t i = 0; i < 3; i++ )
        {
            parameters[par] = angle;
            ++par;
        }
    }
    
    for ( std::size_t i = 0; i < Dimension; i++ )
    {
        parameters[i + par] = ( double ) i * ( double ) transformIndex + ( double ) transformIndex;
    }
}

//------------------------------------------------------------------------------
template< typename SimilarityTransformType >
void
DefineSimilarityParameters( const std::size_t transformIndex,
                            typename SimilarityTransformType::ParametersType &parameters )
{
    const std::size_t Dimension = SimilarityTransformType::InputSpaceDimension;
    
    // Setup parameters
    // 2D: 2 translation, angle 1, scale 1
    // 3D: 3 translation, angle 3, scale 1
    parameters.SetSize( SimilarityTransformType::ParametersDimension );
    
    // Scale, Angle
    const double scale = (( double ) transformIndex + 1.0) * 0.05 + 1.0;
    const double angle = ( double ) transformIndex * -0.06;
    
    if ( Dimension == 2 )
    {
        // See implementation of Similarity2DTransform::SetParameters()
        parameters[0] = scale;
        parameters[1] = angle;
    }
    else if ( Dimension == 3 )
    {
        // See implementation of Similarity3DTransform::SetParameters()
        for ( std::size_t i = 0; i < Dimension; i++ )
        {
            parameters[i] = angle;
        }
        parameters[6] = scale;
    }
    
    // Translation
    for ( std::size_t i = 0; i < Dimension; i++ )
    {
        parameters[i + Dimension] = -1.0 * (( double ) i * ( double ) transformIndex + ( double ) transformIndex);
    }
}

//------------------------------------------------------------------------------
// This helper function completely set the transform
// We are using ITK elastix transforms:
// ITK transforms:
// TransformType, AffineTransformType, TranslationTransformType,
// BSplineTransformType, EulerTransformType, SimilarityTransformType
// elastix Transforms:
// AdvancedCombinationTransformType, AdvancedAffineTransformType,
// AdvancedTranslationTransformType, AdvancedBSplineTransformType,
// AdvancedEulerTransformType, AdvancedSimilarityTransformType
template< typename TransformType, typename AffineTransformType, typename TranslationTransformType, typename BSplineTransformType, typename EulerTransformType, typename SimilarityTransformType, typename AdvancedCombinationTransformType, typename AdvancedAffineTransformType, typename AdvancedTranslationTransformType, typename AdvancedBSplineTransformType, typename AdvancedEulerTransformType, typename AdvancedSimilarityTransformType, typename InputImageType >
void
SetTransform( const std::size_t transformIndex, const std::string &transformName,
              typename TransformType::Pointer &transform,
              typename AdvancedCombinationTransformType::Pointer &advancedTransform,
              const typename InputImageType::ConstPointer &image,
              std::vector< typename BSplineTransformType::ParametersType > &bsplineParameters,
              const std::string &parametersFileName )
{
    if ( transformName == "Affine" )
    {
        if ( advancedTransform.IsNull() )
        {
            // Create Affine transform
            typename AffineTransformType::Pointer affineTransform = AffineTransformType::New();
            
            // Define and set affine parameters
            typename AffineTransformType::ParametersType parameters;
            DefineAffineParameters< AffineTransformType >( parameters );
            affineTransform->SetParameters( parameters );
            
            transform = affineTransform;
        }
        else
        {
            // Create Advanced Affine transform
            typename AdvancedAffineTransformType::Pointer affineTransform = AdvancedAffineTransformType::New();
            advancedTransform->SetCurrentTransform( affineTransform );
            
            // Define and set advanced affine parameters
            typename AdvancedAffineTransformType::ParametersType parameters;
            DefineAffineParameters< AdvancedAffineTransformType >( parameters );
            affineTransform->SetParameters( parameters );
        }
    }
    else if ( transformName == "Translation" )
    {
        if ( advancedTransform.IsNull() )
        {
            // Create Translation transform
            typename TranslationTransformType::Pointer translationTransform = TranslationTransformType::New();
            
            // Define and set translation parameters
            typename TranslationTransformType::ParametersType parameters;
            DefineTranslationParameters< TranslationTransformType >( transformIndex, parameters );
            translationTransform->SetParameters( parameters );
            
            transform = translationTransform;
        }
        else
        {
            // Create Advanced Translation transform
            typename AdvancedTranslationTransformType::Pointer translationTransform = AdvancedTranslationTransformType::New();
            advancedTransform->SetCurrentTransform( translationTransform );
            
            // Define and set advanced translation parameters
            typename AdvancedTranslationTransformType::ParametersType parameters;
            DefineTranslationParameters< AdvancedTranslationTransformType >( transformIndex, parameters );
            translationTransform->SetParameters( parameters );
        }
    }
    else if ( transformName == "BSpline" )
    {
        const unsigned int Dimension = image->GetImageDimension();
        const typename InputImageType::SpacingType inputSpacing = image->GetSpacing();
        const typename InputImageType::PointType inputOrigin = image->GetOrigin();
        const typename InputImageType::DirectionType inputDirection = image->GetDirection();
        const typename InputImageType::RegionType inputRegion = image->GetBufferedRegion();
        const typename InputImageType::SizeType inputSize = inputRegion.GetSize();
        
        typedef typename BSplineTransformType::MeshSizeType MeshSizeType;
        MeshSizeType gridSize;
        gridSize.Fill( 4 );
        
        typedef typename BSplineTransformType::PhysicalDimensionsType PhysicalDimensionsType;
        PhysicalDimensionsType gridSpacing;
        for ( unsigned int d = 0; d < Dimension; d++ )
        {
            gridSpacing[d] = inputSpacing[d] * (inputSize[d] - 1.0);
        }
        
        if ( advancedTransform.IsNull() )
        {
            // Create BSpline transform
            typename BSplineTransformType::Pointer bsplineTransform = BSplineTransformType::New();
            
            // Set grid properties
            bsplineTransform->SetTransformDomainOrigin( inputOrigin );
            bsplineTransform->SetTransformDomainDirection( inputDirection );
            bsplineTransform->SetTransformDomainPhysicalDimensions( gridSpacing );
            bsplineTransform->SetTransformDomainMeshSize( gridSize );
            
            // Define and set b-spline parameters
            typename BSplineTransformType::ParametersType parameters;
            DefineBSplineParameters< BSplineTransformType >( transformIndex, parameters, bsplineTransform,
                                                             parametersFileName );
            
            // Keep them in memory first by copying to the bsplineParameters
            bsplineParameters.push_back( parameters );
            const std::size_t indexAt = bsplineParameters.size() - 1;
            
            // Do not set parameters, the will be destroyed going out of scope
            // instead, set the ones from the bsplineParameters array
            bsplineTransform->SetParameters( bsplineParameters[indexAt] );
            
            transform = bsplineTransform;
        }
        else
        {
            // Create Advanced BSpline transform
            typename AdvancedBSplineTransformType::Pointer bsplineTransform = AdvancedBSplineTransformType::New();
            advancedTransform->SetCurrentTransform( bsplineTransform );
            
            // Set grid properties
            bsplineTransform->SetGridOrigin( inputOrigin );
            bsplineTransform->SetGridDirection( inputDirection );
            bsplineTransform->SetGridSpacing( gridSpacing );
            bsplineTransform->SetGridRegion( gridSize );
            
            // Define and set b-spline parameters
            typename AdvancedBSplineTransformType::ParametersType parameters;
            DefineBSplineParameters< AdvancedBSplineTransformType >( transformIndex, parameters, bsplineTransform,
                                                                     parametersFileName );
            
            // Keep them in memory first by copying to the bsplineParameters
            bsplineParameters.push_back( parameters );
            const std::size_t indexAt = bsplineParameters.size() - 1;
            
            // Do not set parameters, the will be destroyed going out of scope
            // instead, set the ones from the bsplineParameters array
            bsplineTransform->SetParameters( bsplineParameters[indexAt] );
        }
    }
    else if ( transformName == "Euler" )
    {
        // Compute center
        const typename InputImageType::PointType center = ComputeCenterOfTheImage< InputImageType >( image );
        
        if ( advancedTransform.IsNull() )
        {
            // Create Euler transform
            typename EulerTransformType::Pointer eulerTransform = EulerTransformType::New();
            
            // Set center
            eulerTransform->SetCenter( center );
            
            // Define and set euler parameters
            typename EulerTransformType::ParametersType parameters;
            DefineEulerParameters< EulerTransformType >( transformIndex, parameters );
            eulerTransform->SetParameters( parameters );
            
            transform = eulerTransform;
        }
        else
        {
            // Create Advanced Euler transform
            typename AdvancedEulerTransformType::Pointer eulerTransform = AdvancedEulerTransformType::New();
            advancedTransform->SetCurrentTransform( eulerTransform );
            
            // Set center
            eulerTransform->SetCenter( center );
            
            // Define and set advanced euler parameters
            typename AdvancedEulerTransformType::ParametersType parameters;
            DefineEulerParameters< AdvancedEulerTransformType >( transformIndex, parameters );
            eulerTransform->SetParameters( parameters );
        }
    }
    else if ( transformName == "Similarity" )
    {
        // Compute center
        const typename InputImageType::PointType center = ComputeCenterOfTheImage< InputImageType >( image );
        
        if ( advancedTransform.IsNull() )
        {
            // Create Similarity transform
            typename SimilarityTransformType::Pointer similarityTransform = SimilarityTransformType::New();
            
            // Set center
            similarityTransform->SetCenter( center );
            
            // Define and set similarity parameters
            typename SimilarityTransformType::ParametersType parameters;
            DefineSimilarityParameters< SimilarityTransformType >( transformIndex, parameters );
            similarityTransform->SetParameters( parameters );
            
            transform = similarityTransform;
        }
        else
        {
            // Create Advanced Similarity transform
            typename AdvancedSimilarityTransformType::Pointer similarityTransform = AdvancedSimilarityTransformType::New();
            advancedTransform->SetCurrentTransform( similarityTransform );
            
            // Set center
            similarityTransform->SetCenter( center );
            
            // Define and set advanced similarity parameters
            typename AdvancedSimilarityTransformType::ParametersType parameters;
            DefineSimilarityParameters< AdvancedSimilarityTransformType >( transformIndex, parameters );
            similarityTransform->SetParameters( parameters );
        }
    }
}

void
RegisterGPUFactories()
{
    typedef typelist::MakeTypeList< float >::Type OCLImageTypes;
    
    // Register object factory for GPU image and filter
    // All these filters that are constructed after this point are
    // turned into a GPU filter.
    itk::GPUImageFactory2< OCLImageTypes, OCLImageDims >::RegisterOneFactory();
    itk::GPUResampleImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >::RegisterOneFactory();
    itk::GPUCastImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >::RegisterOneFactory();
    
    // Transforms factory registration
    itk::GPUAffineTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUTranslationTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUBSplineTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUEuler3DTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUSimilarity3DTransformFactory2< OCLImageDims >::RegisterOneFactory();
    
    // Interpolators factory registration
    itk::GPUNearestNeighborInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >::RegisterOneFactory();
    itk::GPULinearInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >::RegisterOneFactory();
    itk::GPUBSplineInterpolateImageFunctionFactory2< OCLImageTypes, OCLImageDims >::RegisterOneFactory();
    itk::GPUBSplineDecompositionImageFilterFactory2< OCLImageTypes, OCLImageTypes, OCLImageDims >::RegisterOneFactory();
    
    // Advanced transforms factory registration
    itk::GPUAdvancedCombinationTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUAdvancedMatrixOffsetTransformBaseFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUAdvancedTranslationTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUAdvancedBSplineDeformableTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUAdvancedEuler3DTransformFactory2< OCLImageDims >::RegisterOneFactory();
    itk::GPUAdvancedSimilarity3DTransformFactory2< OCLImageDims >::RegisterOneFactory();
}

void
UnRegisterGPUFactories()
{
    std::cout << "UnRegisterGPUFactories not implemented" << std::endl;
}

}

#endif // end #ifndef __itkTestHelper_h
