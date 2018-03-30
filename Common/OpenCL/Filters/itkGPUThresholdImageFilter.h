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
#ifndef itkCLGPUThresholdImageFilter_h
#define itkCLGPUThresholdImageFilter_h

#include "itkGPUImage.h"
#include "itkGPUBinaryMorphologyImageFilter.h"

#include "itkOpenCLUtil.h"
#include "itkGPUFunctorBase.h"
#include "itkOpenCLKernelManager.h"

// #include "itkGPUKernelManager.h"
#include "itkThresholdImageFilter.h"
#include "itkGPUUnaryFunctorImageFilter.h"

namespace itk
{

namespace Functor
{
template< typename TInput >
class /* ITK_TEMPLATE_EXPORT */ GPUThreshold : public GPUFunctorBase
{
public:
  GPUThreshold()
  {
    m_LowerThreshold = NumericTraits< TInput >::NonpositiveMin();
    m_UpperThreshold = NumericTraits< TInput >::max();
    m_OutsideValue   = NumericTraits< TInput >::ZeroValue();
//    m_InsideValue    = NumericTraits< TOutput >::max();
  }

  ~GPUThreshold() {
  }

  void SetLowerThreshold(const TInput & thresh)
  {
    m_LowerThreshold = thresh;
  }
  void SetUpperThreshold(const TInput & thresh)
  {
    m_UpperThreshold = thresh;
  }
//  void SetInsideValue(const TOutput & value)
//  {
//    m_InsideValue = value;
//  }
  void SetOutsideValue(const TInput & value)
  {
    m_OutsideValue = value;
  }

  /** Setup GPU kernel arguments for this functor.
   * Returns current argument index to set additional arguments in the GPU kernel */
  // int SetGPUKernelArguments(GPUKernelManager::Pointer KernelManager, int KernelHandle)
  int SetGPUKernelArguments(OpenCLKernelManager::Pointer KernelManager, int KernelHandle)
  {
    KernelManager->SetKernelArg(KernelHandle, 0, sizeof(TInput), &(m_LowerThreshold) );
    KernelManager->SetKernelArg(KernelHandle, 1, sizeof(TInput), &(m_UpperThreshold) );
    KernelManager->SetKernelArg(KernelHandle, 2, sizeof(TInput), &(m_OutsideValue) );
//    KernelManager->SetKernelArg(KernelHandle, 2, sizeof(TOutput), &(m_InsideValue) );
//    KernelManager->SetKernelArg(KernelHandle, 3, sizeof(TOutput), &(m_OutsideValue) );
    return 3;
  }

private:
  TInput  m_LowerThreshold;
  TInput  m_UpperThreshold;
//  TOutput m_InsideValue;
  TInput m_OutsideValue;
};
} // end of namespace Functor

/** Create a helper GPU Kernel class for GPUThresholdImageFilter */
itkGPUKernelClassMacro(GPUThresholdImageFilterKernel);

/**
 * \class GPUThresholdImageFilter
 *
 * \brief GPU version of threshold image filter.
 *
 * \ingroup ITKGPUThresholding
 */
template< typename TInputImage >
class /* ITK_TEMPLATE_EXPORT */ GPUThresholdImageFilter :
  public
  GPUUnaryFunctorImageFilter< TInputImage, TInputImage,
                              Functor::GPUThreshold<typename TInputImage::PixelType >,
                              ThresholdImageFilter<TInputImage> >
{
public:
  /** Standard class typedefs. */
  typedef GPUThresholdImageFilter Self;
  typedef GPUUnaryFunctorImageFilter< TInputImage, TInputImage, 
                                      Functor::GPUThreshold< typename TInputImage::PixelType >,
                                      ThresholdImageFilter<TInputImage> > GPUSuperclass;
  typedef ThresholdImageFilter<TInputImage>                     CPUSuperclass;
  typedef SmartPointer< Self >                                  Pointer;
  typedef SmartPointer< const Self >                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUThresholdImageFilter, GPUUnaryFunctorImageFilter);

  /** Pixel types. */
  typedef typename TInputImage::PixelType InputPixelType;
  typedef typename TInputImage::PixelType OutputPixelType;

  /** Type of DataObjects to use for scalar inputs */
  // typedef SimpleDataObjectDecorator< InputPixelType > InputPixelObjectType;

  /** Get OpenCL Kernel source as a string, creates a GetOpenCLSource method */
  itkGetOpenCLSourceFromKernelMacro(GPUThresholdImageFilterKernel);

protected:
  GPUThresholdImageFilter();
  virtual ~GPUThresholdImageFilter() {
  }

  /** This method is used to set the state of the filter before
   * multi-threading. */
  //virtual void BeforeThreadedGenerateData();

  /** Unlike CPU version, GPU version of threshold filter is not
    multi-threaded */
  virtual void GPUGenerateData() ITK_OVERRIDE;

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(GPUThresholdImageFilter);

};

/**
 * \class GPUThresholdImageFilterFactory
 * Object Factory implemenatation for GPUThresholdImageFilter
 *
 * \ingroup ITKGPUThresholding
 */
class GPUThresholdImageFilterFactory : public ObjectFactoryBase
{
public:
  typedef GPUThresholdImageFilterFactory Self;
  typedef ObjectFactoryBase                    Superclass;
  typedef SmartPointer<Self>                   Pointer;
  typedef SmartPointer<const Self>             ConstPointer;

  /** Class methods used to interface with the registered factories. */
  virtual const char* GetITKSourceVersion() const ITK_OVERRIDE
    {
    return ITK_SOURCE_VERSION;
    }
  const char* GetDescription() const ITK_OVERRIDE
    {
    return "A Factory for GPUThresholdImageFilter";
    }

  /** Method for class instantiation. */
  itkFactorylessNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GPUThresholdImageFilterFactory, itk::ObjectFactoryBase);

  /** Register one factory of this type  */
  static void RegisterOneFactory(void)
  {
    GPUThresholdImageFilterFactory::Pointer factory = GPUThresholdImageFilterFactory::New();

    itk::ObjectFactoryBase::RegisterFactory(factory);
  }

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(GPUThresholdImageFilterFactory);

#define OverrideThresholdFilterTypeMacro(ipt,opt,dm) \
    { \
    typedef itk::Image<ipt,dm> InputImageType; \
    typedef itk::Image<opt,dm> OutputImageType; \
    this->RegisterOverride( \
      typeid(itk::ThresholdImageFilter<InputImageType>).name(), \
      typeid(itk::GPUThresholdImageFilter<InputImageType>).name(), \
      "GPU Threshold Image Filter Override", \
      true, \
      itk::CreateObjectFunction<GPUThresholdImageFilter<InputImageType> >::New() ); \
    }

  GPUThresholdImageFilterFactory()
  {
    if( IsGPUAvailable() )
      {
      OverrideThresholdFilterTypeMacro(unsigned char, unsigned char, 1);
      OverrideThresholdFilterTypeMacro(char, char, 1);
      OverrideThresholdFilterTypeMacro(float,float,1);
      OverrideThresholdFilterTypeMacro(int,int,1);
      OverrideThresholdFilterTypeMacro(unsigned int,unsigned int,1);
      OverrideThresholdFilterTypeMacro(double,double,1);

      OverrideThresholdFilterTypeMacro(unsigned char, unsigned char, 2);
      OverrideThresholdFilterTypeMacro(char, char, 2);
      OverrideThresholdFilterTypeMacro(float,float,2);
      OverrideThresholdFilterTypeMacro(int,int,2);
      OverrideThresholdFilterTypeMacro(unsigned int,unsigned int,2);
      OverrideThresholdFilterTypeMacro(double,double,2);

      OverrideThresholdFilterTypeMacro(unsigned char, unsigned char, 3);
      OverrideThresholdFilterTypeMacro(unsigned short, unsigned short, 3);
      OverrideThresholdFilterTypeMacro(char, char, 3);
      OverrideThresholdFilterTypeMacro(float,float,3);
      OverrideThresholdFilterTypeMacro(int,int,3);
      OverrideThresholdFilterTypeMacro(unsigned int,unsigned int,3);
      OverrideThresholdFilterTypeMacro(double,double,3);
      }
  }
#undef OverrideThresholdFilterTypeMacro

};

} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUThresholdImageFilter.hxx"
#endif

#endif
