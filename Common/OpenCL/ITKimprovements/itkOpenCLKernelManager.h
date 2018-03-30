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
#ifndef __itkOpenCLKernelManager_h
#define __itkOpenCLKernelManager_h

#include <itkLightObject.h>
#include <itkObjectFactory.h>

#include "itkGPUDataManager.h"
#include "itkOpenCLContext.h"
#include "itkOpenCLKernel.h"

#include <vector>

namespace itk
{
/** \class OpenCLKernelManager
 * \brief OpenCL kernel manager implemented using OpenCL.
 *
 * This class is responsible for managing the GPU kernel and
 * command queue.
 *
 * \note This file was taken from ITK 4.1.0.
 * It was modified by Denis P. Shamonin and Marius Staring.
 * Division of Image Processing,
 * Department of Radiology, Leiden, The Netherlands.
 * Added functionality is described in the Insight Journal paper:
 * http://hdl.handle.net/10380/3393
 *
 * \ingroup OpenCL
 */

class OpenCLContext;

class ITKOpenCL_EXPORT OpenCLKernelManager : public LightObject
{
public:

  /** Standard class typedefs. */
  typedef OpenCLKernelManager        Self;
  typedef LightObject                Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( OpenCLKernelManager, LightObject );

  /** Returns the OpenCL context that this kernel manager was created within. */
  OpenCLContext * GetContext() const { return this->m_Context; }

  /** Returns the . */
  OpenCLKernel & GetKernel( const std::size_t kernelId );

  OpenCLEvent LaunchKernel( const std::size_t kernelId );

  OpenCLEvent LaunchKernel( const std::size_t kernelId,
    const OpenCLSize & global_work_size,
    const OpenCLSize & local_work_size = OpenCLSize::null,
    const OpenCLSize & global_work_offset = OpenCLSize::null );

  OpenCLEvent LaunchKernel( const std::size_t kernelId, const OpenCLEventList & event_list );

  OpenCLEvent LaunchKernel( const std::size_t kernelId,
    const OpenCLEventList & event_list,
    const OpenCLSize & global_work_size,
    const OpenCLSize & local_work_size = OpenCLSize::null,
    const OpenCLSize & global_work_offset = OpenCLSize::null );

  std::size_t CreateKernel( const OpenCLProgram & program, const std::string & name );

  OpenCLProgram BuildProgramFromSourceCode( const std::string & sourceCode,
    const std::string & prefixSourceCode = std::string(),
    const std::string & postfixSourceCode = std::string(),
    const std::string & extraBuildOptions = std::string() );

  OpenCLProgram BuildProgramFromSourceFile( const std::string & fileName,
    const std::string & prefixSourceCode = std::string(),
    const std::string & postfixSourceCode = std::string(),
    const std::string & extraBuildOptions = std::string() );

  /** Sets the global work size for all instances of the kernels to \a size.
   * \sa SetLocalWorkSizeForAllKernels(), SetGlobalWorkOffsetForAllKernels() */
  void SetGlobalWorkSizeForAllKernels( const OpenCLSize & size );

  /** Sets the local work size for all instances of the kernels to \a size.
   * \sa SetGlobalWorkSizeForAllKernels(), SetGlobalWorkOffsetForAllKernels() */
  void SetLocalWorkSizeForAllKernels( const OpenCLSize & size );

  /** Sets the offset for all instances of the kernels to \a offset.
   * \sa SetGlobalWorkSizeForAllKernels(), SetLocalWorkSizeForAllKernels() */
  void SetGlobalWorkOffsetForAllKernels( const OpenCLSize & offset );

  bool SetKernelArg( const std::size_t kernelId,
    const cl_uint argId, const std::size_t argSize, const void * argVal );

  bool SetKernelArgForAllKernels( const cl_uint argId,
    const std::size_t argSize, const void * argVal );

  bool SetKernelArgWithImage( const std::size_t kernelId, cl_uint argId, const GPUDataManager::Pointer manager );
  
  /** Pass to GPU both the pixel buffer and the buffered region. */
  //template< typename TGPUImageDataManager >
  //bool SetKernelArgWithImageAndBufferedRegion(int kernelIdx, cl_uint &argIdx, typename TGPUImageDataManager::Pointer manager);
  template< typename TGPUImageDataManager >
  bool SetKernelArgWithImageAndBufferedRegion( int kernelIdx, cl_uint& argIdx, TGPUImageDataManager* manager )
  {
    // if(kernelIdx < 0 || kernelIdx >= (int)/* m_KernelContainer */m_Kernels.size() ) return false;
    if( kernelIdx < 0 || kernelIdx >= this->m_Kernels.size() ) { return false; }

    cl_int errid;

    errid = clSetKernelArg( this->GetKernel( kernelIdx ).GetKernelId(), argIdx, sizeof(cl_mem),
      manager->GetGPUBufferPointer() );
    // OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
    this->m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );

    m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
    m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager;
    argIdx++;

    //this->SetKernelArg(kernelIdx, argIdx++, sizeof(int), &(TGPUImageDataManager::ImageDimension) );

    //the starting index for the buffered region
    errid = clSetKernelArg(/* m_KernelContainer */this->GetKernel( kernelIdx ).GetKernelId(), argIdx, sizeof(cl_mem),
      manager->GetGPUBufferedRegionIndex()->GetGPUBufferPointer() );
    // OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
    this->m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );

    m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
    m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager->GetGPUBufferedRegionIndex();
    argIdx++;

    //the size for the buffered region
    errid = clSetKernelArg(/* m_KernelContainer */this->GetKernel( kernelIdx ).GetKernelId(), argIdx, sizeof(cl_mem),
      manager->GetGPUBufferedRegionSize()->GetGPUBufferPointer() );
    // OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
    this->m_Context->ReportError( errid, __FILE__, __LINE__, ITK_LOCATION );

    m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
    m_KernelArgumentReady[kernelIdx][argIdx].m_GPUDataManager = manager->GetGPUBufferedRegionSize();
    argIdx++;

    return true;
  }

protected:

  OpenCLKernelManager();
  virtual ~OpenCLKernelManager();

  bool CheckArgumentReady( const std::size_t kernelId );

  void ResetArguments( const std::size_t kernelIdx );

private:

  OpenCLKernelManager( const Self & );   // purposely not implemented
  void operator=( const Self & );        // purposely not implemented

  OpenCLContext * m_Context;

  struct KernelArgumentList
  {
    bool                    m_IsReady;
    GPUDataManager::Pointer m_GPUDataManager;
  };

  std::vector< OpenCLKernel >                      m_Kernels;
  std::vector< std::vector< KernelArgumentList > > m_KernelArgumentReady;
};

} // end namespace itk

#endif /* __itkOpenCLKernelManager_h */
