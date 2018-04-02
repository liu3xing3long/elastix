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
#ifndef itkGPUNeighborhoodOperatorImageFilter_hxx
#define itkGPUNeighborhoodOperatorImageFilter_hxx

#include "itkNeighborhoodAlgorithm.h"
#include "itkNeighborhoodInnerProduct.h"
#include "itkImageRegionIterator.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkProgressReporter.h"
#include "itkGPUNeighborhoodOperatorImageFilter.h"

#include "itkOpenCLUtil.h"
#include "itkOpenCLKernelToImageBridge.h"

//#define FILTER_DEBUG


namespace itk
{
/*
template< typename TInputImage, typename TOutputImage, typename TOperatorValueType >
void
GPUNeighborhoodOperatorImageFilter< TInputImage, TOutputImage, TOperatorValueType >
::GenerateInputRequestedRegion()
{
  // call the superclass' implementation of this method. this should
  // copy the output requested region to the input requested region
  Superclass::GenerateInputRequestedRegion();

  // get pointers to the input and output
  InputImagePointer inputPtr =
    const_cast< TInputImage * >( this->GetInput() );

  if ( !inputPtr )
    {
    return;
    }

  // get a copy of the input requested region (should equal the output
  // requested region)
  typename TInputImage::RegionType inputRequestedRegion;
  inputRequestedRegion = inputPtr->GetRequestedRegion();

  // pad the input requested region by the operator radius
  inputRequestedRegion.PadByRadius( m_Operator.GetRadius() );

  // crop the input requested region at the input's largest possible region
  if ( inputRequestedRegion.Crop( inputPtr->GetLargestPossibleRegion() ) )
    {
    inputPtr->SetRequestedRegion(inputRequestedRegion);
    return;
    }
  else
    {
    // Couldn't crop the region (requested region is outside the largest
    // possible region).  Throw an exception.

    // store what we tried to request (prior to trying to crop)
    inputPtr->SetRequestedRegion(inputRequestedRegion);

    // build an exception
    InvalidRequestedRegionError e(__FILE__, __LINE__);
    e.SetLocation(ITK_LOCATION);
    e.SetDescription("Requested region is (at least partially) outside the largest possible region.");
    e.SetDataObject(inputPtr);
    throw e;
    }
}
*/

template< typename TInputImage, typename TOutputImage, typename TOperatorValueType, typename TParentImageFilter >
GPUNeighborhoodOperatorImageFilter< TInputImage, TOutputImage, TOperatorValueType, TParentImageFilter >::GPUNeighborhoodOperatorImageFilter()
{
    // Create GPU buffer to store neighborhood coefficient.
    // This will be used as __constant memory in the GPU kernel.
    m_NeighborhoodGPUBuffer = NeighborhoodGPUBufferType::New();
    
    std::ostringstream defines;
    
    if ( TInputImage::ImageDimension > 3 || TInputImage::ImageDimension < 1 )
    {
        itkExceptionMacro( "GPUneighborhoodOperatorImageFilter supports 1/2/3D image." );
    }
    
    defines << "#define DIM_" << TInputImage::ImageDimension << "\n";
    
    defines << "#define INTYPE ";
    GetTypenameInString( typeid( typename TInputImage::PixelType ), defines );
    
    defines << "#define OUTTYPE ";
    GetTypenameInString( typeid( typename TOutputImage::PixelType ), defines );
    
    defines << "#define OPTYPE ";
    GetTypenameInString( typeid( TOperatorValueType ), defines );
    
    // std::cout << "Defines: " << defines.str() << std::endl;
    itkDebugMacro( << "Defines: " << defines.str() << std::endl );
    
    const char *GPUSource = GPUNeighborhoodOperatorImageFilter::GetOpenCLSource();
    
    // load and build program
    // this->m_GPUKernelManager->LoadProgramFromString( GPUSource, defines.str().c_str() );
    const OpenCLProgram program = this->m_GPUKernelManager->BuildProgramFromSourceCode( GPUSource, defines.str().c_str() );
    
    // create kernel
    m_NeighborhoodOperatorFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel( program, "NeighborOperatorFilter" );
}

template< typename TInputImage, typename TOutputImage, typename TOperatorValueType, typename TParentImageFilter >
void
GPUNeighborhoodOperatorImageFilter< TInputImage, TOutputImage, TOperatorValueType, TParentImageFilter >::SetOperator(
        const OutputNeighborhoodType &p )
{
    /** Call CPU SetOperator */
    CPUSuperclass::SetOperator( p );
    
    /** Create GPU memory for operator coefficients */
    m_NeighborhoodGPUBuffer->Initialize();
    
    typename NeighborhoodGPUBufferType::IndexType index;
    typename NeighborhoodGPUBufferType::SizeType size;
    typename NeighborhoodGPUBufferType::RegionType region;
    
    for ( unsigned int i = 0; i < ImageDimension; i++ )
    {
        index[i] = 0;
        size[i] = ( unsigned int ) (p.GetSize( i ));
    }
    region.SetSize( size );
    region.SetIndex( index );
    
    m_NeighborhoodGPUBuffer->SetRegions( region );
    m_NeighborhoodGPUBuffer->Allocate();
    
    /** Copy coefficients */
    ImageRegionIterator< NeighborhoodGPUBufferType > iit( m_NeighborhoodGPUBuffer, m_NeighborhoodGPUBuffer->GetLargestPossibleRegion() );
    
    typename OutputNeighborhoodType::ConstIterator nit = p.Begin();
    
    for ( iit.GoToBegin(); !iit.IsAtEnd(); ++iit, ++nit )
    {
        iit.Set( static_cast< typename NeighborhoodGPUBufferType::PixelType >( *nit ) );
    }
    
    /** Mark GPU dirty */
    m_NeighborhoodGPUBuffer->GetGPUDataManager()->SetGPUBufferDirty();
}

template< typename TInputImage, typename TOutputImage, typename TOperatorValueType, typename TParentImageFilter >
void
GPUNeighborhoodOperatorImageFilter< TInputImage, TOutputImage, TOperatorValueType, TParentImageFilter >::GPUGenerateData()
{
    int kHd = m_NeighborhoodOperatorFilterGPUKernelHandle;
    
    typedef typename itk::GPUTraits< TInputImage >::Type GPUInputImage;
    typedef typename itk::GPUTraits< TOutputImage >::Type GPUOutputImage;
    typedef GPUImageDataManager< GPUInputImage > GPUInputManagerType;
    typedef GPUImageDataManager< GPUOutputImage > GPUOutputManagerType;
    
    typename GPUInputImage::Pointer inPtr = dynamic_cast< GPUInputImage * >( this->ProcessObject::GetInput( 0 ) );
    typename GPUOutputImage::Pointer otPtr = dynamic_cast< GPUOutputImage * >( this->ProcessObject::GetOutput( 0 ) );
    
    //typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
    typename GPUOutputImage::SizeType outSize = otPtr->GetBufferedRegion().GetSize();
    
    int radius[3];
    int imgSize[3];
    
    radius[0] = radius[1] = radius[2] = 0;
    imgSize[0] = imgSize[1] = imgSize[2] = 1;
    
    int ImageDim = ( int ) TInputImage::ImageDimension;
    
    for ( int i = 0; i < ImageDim; i++ )
    {
        radius[i] = (this->GetOperator()).GetRadius( i );
        imgSize[i] = outSize[i];
    }
    
    typename TInputImage::SizeType localSize, globalSize;
    // localSize[0] = localSize[1] = localSize[2] = OpenCLGetLocalBlockSize(ImageDim);
    for ( int i = 0; i < ImageDim; i++ )
    {
        localSize[i] = OpenCLGetLocalBlockSize( ImageDim );
    }
    
    for ( int i = 0; i < ImageDim; i++ )
    {
        globalSize[i] = localSize[i] * ( unsigned int ) ceil( ( float ) outSize[i] / ( float ) localSize[i] ); //
        // total
        // #
        // of
        // threads
    }
    
#ifdef FILTER_DEBUG
    std::cout << "image size" << std::endl;
    for ( int i = 0; i < ImageDim; i++ )
        std::cout << imgSize[i] << " ";
    std::cout << std::endl;
    
    std::cout << "radius size" << std::endl;
    for ( int i = 0; i < ImageDim; i++ )
        std::cout << radius[i] << " ";
    std::cout << std::endl;
#endif
    
    // arguments set up
    cl_uint argidx = 0;
    
    //
    // SHOULD BE SUPER CAREFULL WITH
    // ALL THESE CAST !!!!
    //
    typename GPUInputManagerType::Pointer pInImageManager = itkDynamicCastInDebugMode< GPUInputManagerType * >( inPtr->GetGPUDataManager().GetPointer() );
    typename GPUOutputManagerType::Pointer pOutImageManager = itkDynamicCastInDebugMode< GPUOutputManagerType * >( otPtr->GetGPUDataManager().GetPointer() );
    
    //
    // Set GPU Image and buffer region
    // Here, we split the function SetKernelArgWithImageAndBufferedRegion SEPARATE
    //

#ifdef FILTER_DEBUG
    std::cout << "setting input image and region" << std::endl;
#endif

    this->m_GPUKernelManager->SetKernelArgWithImage( kHd, argidx++, inPtr->GetGPUDataManager() );
    this->m_GPUKernelManager->SetKernelArgWithImage( kHd, argidx++, pInImageManager->GetGPUBufferedRegionIndex() );
    this->m_GPUKernelManager->SetKernelArgWithImage( kHd, argidx++, pInImageManager->GetGPUBufferedRegionSize() );


#ifdef FILTER_DEBUG
    std::cout << "setting output image and region" << std::endl;
#endif
    // do the same to the output image
    this->m_GPUKernelManager->SetKernelArgWithImage( kHd, argidx++, otPtr->GetGPUDataManager() );
    this->m_GPUKernelManager->SetKernelArgWithImage( kHd, argidx++, pOutImageManager->GetGPUBufferedRegionIndex() );
    this->m_GPUKernelManager->SetKernelArgWithImage( kHd, argidx++, pOutImageManager->GetGPUBufferedRegionSize() );

#ifdef FILTER_DEBUG
    std::cout << "setting operator buffer" << std::endl;
#endif
    // finally, we set the neighourhood GPU buffer
    this->m_GPUKernelManager->SetKernelArgWithImage( kHd, argidx++, this->m_NeighborhoodGPUBuffer->GetGPUDataManager() );


#ifdef FILTER_DEBUG
    std::cout << "setting radius" << std::endl;
#endif
    for ( int i = 0; i < ( int ) TInputImage::ImageDimension; i++ )
    {
        this->m_GPUKernelManager->SetKernelArg( kHd, argidx++, sizeof( int ), &(radius[i]) );
    }
    
    //for(int i=0; i<(int)TInputImage::ImageDimension; i++)
    //  {
    //  this->m_GPUKernelManager->SetKernelArg(kHd, argidx++, sizeof(int), &(imgSize[i]) );
    //  }
    
    // launch kernel
    //  OpenCLEvent event =  this->m_GPUKernelManager->LaunchKernel(kHd, ImageDim,
    //                                         OpenCLSize(globalSize),
    //                                         OpenCLSize(localSize));

#ifdef FILTER_DEBUG
    std::cout << "lauching kernel " << std::endl;
#endif
    OpenCLEvent event = this->m_GPUKernelManager->LaunchKernel( kHd, /*ImageDim,*/
                                                                OpenCLSize( globalSize ), OpenCLSize( localSize ) );
    
    event.WaitForFinished();
}

} // end namespace itk

#endif


#undef FILTER_DEBUG