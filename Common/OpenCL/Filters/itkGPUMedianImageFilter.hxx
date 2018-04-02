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
#ifndef itkCLGPUMedianImageFilter_hxx
#define itkCLGPUMedianImageFilter_hxx

#include "itkGPUMedianImageFilter.h"
#include "itkOpenCLProfilingTimeProbe.h"

#define FILTER_DEBUG


namespace itk
{

/*
 * MAX_3D_RADIUS = 32, int = 4B, ( 2 * 32 + 1 )^3 = 274625
 * total = 1MB, if kernels = 3092, total = 3 GB
 * should not exceed half of 8 GB (some cards 12 GB) GPU memory
 */

template< typename TInputImage, typename TOutputImage >
GPUMedianImageFilter< TInputImage, TOutputImage >::GPUMedianImageFilter()
{
    std::ostringstream defines;
    
    if ( TInputImage::ImageDimension > 3 )
    {
        itkExceptionMacro( "GPUMedianImageFilter supports 1/2/3D image." );
    }
    
    defines << "#define DIM_" << TInputImage::ImageDimension << "\n";
    defines << "#define PIXELTYPE ";
    GetTypenameInString( typeid( typename TInputImage::PixelType ), defines );
    
    // Define m_DeviceLocalMemorySize as:
    // rad_x * rad_y * rad_z
    unsigned int ImageDim = ( unsigned int ) TInputImage::ImageDimension;
    const unsigned long localMemSize = this->m_GPUKernelManager->GetContext()->GetDefaultDevice().GetLocalMemorySize();
    this->m_DeviceLocalMemorySize = (localMemSize) / sizeof( float );
    
    //    defines << "#define BUFFSIZE " << this->m_DeviceLocalMemorySize << "\n";

#ifdef FILTER_DEBUG
    char debug_msg[4096];
    sprintf( debug_msg, "local memory size %lu, support %u array", localMemSize,
             ( unsigned int ) this->m_DeviceLocalMemorySize );
    std::cout << debug_msg << std::endl;
#endif
    
    itkDebugMacro( << "Defines: " << defines.str() << std::endl );
    
    const char *GPUSource = GPUMedianImageFilter::GetOpenCLSource();
    
    // load and build program
    // this->m_GPUKernelManager->LoadProgramFromString( GPUSource, defines.str().c_str() );
    const OpenCLProgram program = this->m_GPUKernelManager->BuildProgramFromSourceCode( GPUSource,
                                                                                        defines.str().c_str() );
    
    if ( program.IsNull() )
    {
        itkExceptionMacro( << "Kernel has not been loaded from string:\n"
                                   << defines.str() << std::endl << GPUSource );
    }
    
    // create kernel
    m_MedianFilterGPUKernelHandle = this->m_GPUKernelManager->CreateKernel( program, "MedianFilter" );
}

template< typename TInputImage, typename TOutputImage >
GPUMedianImageFilter< TInputImage, TOutputImage >::~GPUMedianImageFilter()
{

}

template< typename TInputImage, typename TOutputImage >
void
GPUMedianImageFilter< TInputImage, TOutputImage >::PrintSelf( std::ostream &os, Indent indent ) const
{
    Superclass::PrintSelf( os, indent );
}

template< typename TInputImage, typename TOutputImage >
void
GPUMedianImageFilter< TInputImage, TOutputImage >::GPUGenerateData()
{

#ifdef FILTER_DEBUG
    itk::OpenCLProfilingTimeProbe timer( "Creating OpenCL program using clCreateProgramWithSource" );
#endif
    
    typedef typename itk::GPUTraits< TInputImage >::Type GPUInputImage;
    typedef typename itk::GPUTraits< TOutputImage >::Type GPUOutputImage;
    
    typename GPUInputImage::Pointer inPtr = dynamic_cast< GPUInputImage * >( this->ProcessObject::GetInput( 0 ) );
    typename GPUOutputImage::Pointer otPtr = dynamic_cast< GPUOutputImage * >( this->ProcessObject::GetOutput( 0 ) );
    
    typename GPUOutputImage::SizeType outSize = otPtr->GetLargestPossibleRegion().GetSize();
    
    int radius[3];
    int imgSize[3];
    
    radius[0] = radius[1] = radius[2] = 0;
    imgSize[0] = imgSize[1] = imgSize[2] = 1;
    
    unsigned int ImageDim = ( unsigned int ) TInputImage::ImageDimension;
    
    for ( int i = 0; i < ImageDim; i++ )
    {
        radius[i] = (this->GetRadius())[i];
        imgSize[i] = outSize[i];
    }
    
    typename GPUInputImage::SizeType localSize, globalSize;
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
    char debug_msg[4096];
    sprintf( debug_msg, "imagesize (%d, %d, %d), radius (%d, %d, %d), localsize (%lu, %lu, %lu), globalsize (%lu, %lu, %lu)",
             imgSize[0], imgSize[1], imgSize[2], radius[0], radius[1], radius[2], localSize[0], localSize[1],
             localSize[2], globalSize[0], globalSize[1], globalSize[2] );
    std::cout << debug_msg << std::endl;
#endif
    
    // arguments set up
    cl_int argidx = 0;
    this->m_GPUKernelManager->SetKernelArgWithImage( m_MedianFilterGPUKernelHandle, argidx++,
                                                     inPtr->GetGPUDataManager() );
    this->m_GPUKernelManager->SetKernelArgWithImage( m_MedianFilterGPUKernelHandle, argidx++,
                                                     otPtr->GetGPUDataManager() );
    
    for ( int i = 0; i < ImageDim; i++ )
    {
        this->m_GPUKernelManager->SetKernelArg( m_MedianFilterGPUKernelHandle, argidx++, sizeof( int ), &(radius[i]) );
    }
    
    for ( int i = 0; i < ImageDim; i++ )
    {
        this->m_GPUKernelManager->SetKernelArg( m_MedianFilterGPUKernelHandle, argidx++, sizeof( int ), &(imgSize[i]) );
    }
    
    // launch kernel
//  OpenCLEvent event = this->m_GPUKernelManager->LaunchKernel( m_MedianFilterGPUKernelHandle, (int)TInputImage::ImageDimension,
//                                          OpenCLSize(globalSize),
//                                          OpenCLSize(localSize) );
    
    OpenCLEvent event = this->m_GPUKernelManager->LaunchKernel(
            m_MedianFilterGPUKernelHandle, /*(int)TInputImage::ImageDimension,*/
            OpenCLSize( globalSize ), OpenCLSize( localSize ) );
    event.WaitForFinished();
    
    itkDebugMacro( << "GPUMedianImageFilter::GPUGenerateData() finished" );
}

} // end namespace itk

#endif
