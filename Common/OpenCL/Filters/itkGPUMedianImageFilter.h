/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkCLGPUMedianImageFilter_h
#define itkCLGPUMedianImageFilter_h

#include "itkMedianImageFilter.h"
#include "itkGPUBoxImageFilter.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"
#include "itkOpenCLUtil.h"

namespace itk
{
/** \class GPUMedianImageFilter
 *
 * \brief GPU-enabled implementation of the GPUMedianImageFilter.
 *
 * Current GPU mean filter reads in neighborhood pixels from global memory.
 *
 * \ingroup ITKGPUSmoothing
 */

/** Create a helper GPU Kernel class for GPUMedianImageFilter */
itkGPUKernelClassMacro( GPUMedianImageFilterKernel );

template< typename TInputImage, typename TOutputImage >
class ITK_TEMPLATE_EXPORT GPUMedianImageFilter: //public GPUImageToImageFilter<
        // TInputImage, TOutputImage,
        // MeanImageFilter< TInputImage,
        // TOutputImage > >
        public GPUBoxImageFilter< TInputImage, TOutputImage, MedianImageFilter< TInputImage, TOutputImage > >
{
public:
    /** Standard class typedefs. */
    typedef GPUMedianImageFilter Self;
    typedef GPUBoxImageFilter< TInputImage, TOutputImage, MedianImageFilter< TInputImage, TOutputImage > > Superclass;
    typedef SmartPointer< Self > Pointer;
    typedef SmartPointer< const Self > ConstPointer;
    
    itkNewMacro( Self );
    
    /** Run-time type information (and related methods). */
    itkTypeMacro( GPUMedianImageFilter, GPUBoxImageFilter );
    
    /** Superclass typedefs. */
    typedef typename Superclass::OutputImageRegionType OutputImageRegionType;
    typedef typename Superclass::OutputImagePixelType OutputImagePixelType;
    
    /** Some convenient typedefs. */
    typedef TInputImage InputImageType;
    typedef typename InputImageType::Pointer InputImagePointer;
    typedef typename InputImageType::ConstPointer InputImageConstPointer;
    typedef typename InputImageType::RegionType InputImageRegionType;
    typedef typename InputImageType::PixelType InputImagePixelType;
    
    /** ImageDimension constants */
    itkStaticConstMacro( InputImageDimension, unsigned int, TInputImage::ImageDimension );
    itkStaticConstMacro( OutputImageDimension, unsigned int, TOutputImage::ImageDimension );
    
    /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
    itkGetOpenCLSourceFromKernelMacro( GPUMedianImageFilterKernel );

protected:
    GPUMedianImageFilter();
    
    ~GPUMedianImageFilter();
    
    virtual void
    PrintSelf( std::ostream &os, Indent indent ) const ITK_OVERRIDE;
    
    virtual void
    GPUGenerateData() ITK_OVERRIDE;

private: ITK_DISALLOW_COPY_AND_ASSIGN( GPUMedianImageFilter );
    
    std::size_t  m_MedianFilterGPUKernelHandle;
    std::size_t m_DeviceLocalMemorySize;
};

/** \class GPUMedianImageFilterFactory
 *
 * \brief Object Factory implemenatation for GPUMedianImageFilter
 * \ingroup ITKGPUSmoothing
 */
class GPUMedianImageFilterFactory:
        public ObjectFactoryBase
{
public:
    typedef GPUMedianImageFilterFactory Self;
    typedef ObjectFactoryBase Superclass;
    typedef SmartPointer< Self > Pointer;
    typedef SmartPointer< const Self > ConstPointer;
    
    /** Class methods used to interface with the registered factories. */
    virtual const char *
    GetITKSourceVersion() const ITK_OVERRIDE
    {
        return ITK_SOURCE_VERSION;
    }
    
    const char *
    GetDescription() const ITK_OVERRIDE
    {
        return "A Factory for GPUMedianImageFilter";
    }
    
    /** Method for class instantiation. */
    itkFactorylessNewMacro( Self );
    
    /** Run-time type information (and related methods). */
    itkTypeMacro( GPUMedianImageFilterFactory, itk::ObjectFactoryBase );
    
    /** Register one factory of this type  */
    static void
    RegisterOneFactory( void )
    {
        GPUMedianImageFilterFactory::Pointer factory = GPUMedianImageFilterFactory::New();
        
        ObjectFactoryBase::RegisterFactory( factory );
    }

private: ITK_DISALLOW_COPY_AND_ASSIGN( GPUMedianImageFilterFactory );

#define OverrideMedianFilterTypeMacro( ipt, opt, dm ) \
    { \
    typedef Image<ipt,dm> InputImageType; \
    typedef Image<opt,dm> OutputImageType; \
    this->RegisterOverride( \
      typeid(MedianImageFilter<InputImageType,OutputImageType>).name(), \
      typeid(GPUMedianImageFilter<InputImageType,OutputImageType>).name(), \
      "GPU Median Image Filter Override", \
      true, \
      CreateObjectFunction<GPUMedianImageFilter<InputImageType,OutputImageType> >::New() ); \
    }
    
    GPUMedianImageFilterFactory()
    {
        if ( IsGPUAvailable() )
        {
            OverrideMedianFilterTypeMacro( unsigned char, unsigned char, 1 );
            OverrideMedianFilterTypeMacro( char, char, 1 );
            OverrideMedianFilterTypeMacro( float, float, 1 );
            OverrideMedianFilterTypeMacro( int, int, 1 );
            OverrideMedianFilterTypeMacro( unsigned int, unsigned int, 1 );
            OverrideMedianFilterTypeMacro( double, double, 1 );
            
            OverrideMedianFilterTypeMacro( unsigned char, unsigned char, 2 );
            OverrideMedianFilterTypeMacro( char, char, 2 );
            OverrideMedianFilterTypeMacro( float, float, 2 );
            OverrideMedianFilterTypeMacro( int, int, 2 );
            OverrideMedianFilterTypeMacro( unsigned int, unsigned int, 2 );
            OverrideMedianFilterTypeMacro( double, double, 2 );
            
            OverrideMedianFilterTypeMacro( unsigned char, unsigned char, 3 );
            OverrideMedianFilterTypeMacro( char, char, 3 );
            OverrideMedianFilterTypeMacro( float, float, 3 );
            OverrideMedianFilterTypeMacro( int, int, 3 );
            OverrideMedianFilterTypeMacro( unsigned int, unsigned int, 3 );
            OverrideMedianFilterTypeMacro( double, double, 3 );
        }
    }
    
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION

#include "itkGPUMedianImageFilter.hxx"

#endif

#endif
