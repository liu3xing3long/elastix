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
#ifndef __itkGPUBinaryErodeImageFilter_hxx
#define __itkGPUBinaryErodeImageFilter_hxx

#include "itkGPUBinaryErodeImageFilter.h"

#include "itkGPUKernelManagerHelperFunctions.h"
#include "itkGPUMath.h"
#include "itkGPUImageBase.h"

#include "itkImageLinearIteratorWithIndex.h"
#include "itkTimeProbe.h"
#include "itkImageRegionSplitterSlowDimension.h"

#include "itkOpenCLUtil.h"
#include "itkOpenCLKernelToImageBridge.h"


namespace itk
{
template< typename TInputImage, typename TOutputImage, typename TKernel >
GPUBinaryErodeImageFilter< TInputImage, TOutputImage, TKernel >
::GPUBinaryErodeImageFilter()
{
  this->m_BoundaryToForeground = true;

  std::ostringstream defines;

  if(TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1)
    {
    itkExceptionMacro("GPUBinaryErodeImageFilter supports 1/2/3D image.");
    }

  defines << "#define DIM_" << TInputImage::ImageDimension << "\n";

  defines << "#define INTYPE ";
  GetTypenameInString( typeid ( typename TInputImage::PixelType ), defines );

  defines << "#define OUTTYPE ";
  GetTypenameInString( typeid ( typename TOutputImage::PixelType ), defines );

  defines << "#define OPTYPE ";
  GetTypenameInString( typeid ( typename TKernel::PixelType ), defines );

  defines << "#define BOOL ";
  GetTypenameInString( typeid(unsigned char), defines );

  itkDebugMacro( << "Defines: " << defines.str() );

  const char* GPUSource = GPUBinaryErodeImageFilter::GetOpenCLSource();

  // load and build program
  const OpenCLProgram program = this->m_GPUKernelManager->BuildProgramFromSourceCode( GPUSource, defines.str().c_str() );
  
  if( program.IsNull() )
  {
    itkExceptionMacro( << "Kernel has not been loaded from string:\n"
                       << defines.str() << std::endl << GPUSource);
  }

  // create kernel
  m_BinaryErodeFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel(program, "BinaryErodeFilter");
}


template< typename TInputImage, typename TOutputImage, typename TKernel >
void
GPUBinaryErodeImageFilter< TInputImage, TOutputImage, TKernel >
::GPUGenerateData()
{
  itkDebugMacro( << "GPUBinaryErodeImageFilter::GPUGenerateData() begin" );
  
  int kHd = m_BinaryErodeFilterGPUKernelHandle;

  typedef typename GPUTraits< TInputImage >::Type  GPUInputImage;
  typedef typename GPUTraits< TOutputImage >::Type GPUOutputImage;
  typedef GPUImageDataManager<GPUInputImage>            GPUInputManagerType;
  typedef GPUImageDataManager<GPUOutputImage>           GPUOutputManagerType;

  typename GPUInputImage::Pointer  inPtr =  dynamic_cast< GPUInputImage * >( this->ProcessObject::GetInput(0) );
  typename GPUOutputImage::Pointer otPtr =  dynamic_cast< GPUOutputImage * >( this->ProcessObject::GetOutput(0) );

  // Perform the safe check
  if( inPtr.IsNull() )
  {
    itkExceptionMacro( << "The GPU InputImage is NULL. Filter unable to perform." );
    return;
  }
  if( otPtr.IsNull() )
  {
    itkExceptionMacro( << "The GPU OutputImage is NULL. Filter unable to perform." );
    return;
  }

  typename GPUOutputImage::SizeType outSize = otPtr->GetBufferedRegion().GetSize();

  int radius[3];
  int imgSize[3];

  radius[0] = radius[1] = radius[2] = 0;
  imgSize[0] = imgSize[1] = imgSize[2] = 1;

  int ImageDim = (int)TInputImage::ImageDimension;

  for(int i=0; i<ImageDim; i++)
  {
    radius[i]  = (this->GetKernel()).GetRadius(i);
    imgSize[i] = outSize[i];
  }

  typename GPUInputImage::SizeType globalSize, localSize;

  for(int i=0; i < ImageDim; i++)
  {
    localSize[i] = OpenCLGetLocalBlockSize(ImageDim);
  } 

  for(int i=0; i<ImageDim; i++)
  {
    globalSize[i] = localSize[i]*(unsigned int)ceil( (float)outSize[i]/(float)localSize[i]);
  }

  // arguments set up
  int argidx = 0;

  typename GPUInputManagerType::Pointer pInImageManager = itkDynamicCastInDebugMode<GPUInputManagerType*>(inPtr->GetGPUDataManager().GetPointer());
  typename GPUOutputManagerType::Pointer pOutImageManager = itkDynamicCastInDebugMode<GPUOutputManagerType*>(otPtr->GetGPUDataManager().GetPointer());

  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, inPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, pInImageManager->GetGPUBufferedRegionIndex());
  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, pInImageManager->GetGPUBufferedRegionSize() );

  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, otPtr->GetGPUDataManager());
  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, pOutImageManager->GetGPUBufferedRegionIndex());
  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, pOutImageManager->GetGPUBufferedRegionSize() );

  this->m_GPUKernelManager->SetKernelArgWithImage(kHd, argidx++, this->m_NeighborhoodGPUBuffer->GetGPUDataManager() );

  for(int i=0; i<(int)TInputImage::ImageDimension; i++)
  {
    // printf("r %d, %d \n", i, radius[i]);
    this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(int), &(radius[i]) );
  }

  this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(int), &(this->m_seCount));
  this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(InputPixelType), &(this->m_ForegroundValue));
  this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(InputPixelType), &(this->m_BackgroundValue));

  unsigned char borderFg = this->m_BoundaryToForeground ? 1 : 0;
  this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(unsigned char), &(borderFg));
  
  // launch kernel
  OpenCLEvent event = this->m_GPUKernelManager->LaunchKernel(kHd,
                                         OpenCLSize(globalSize), 
                                         OpenCLSize(localSize));

  event.WaitForFinished();

  itkDebugMacro( << "GPUBinaryDilateImageFilter::GPUGenerateData() finished" );
}

} // end namespace itk

#endif
