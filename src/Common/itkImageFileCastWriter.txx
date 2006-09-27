/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkImageFileCastWriter_txx
#define _itkImageFileCastWriter_txx

#include "itkImageFileCastWriter.h"
#include "itkDataObject.h"
#include "itkObjectFactoryBase.h"
#include "itkImageIOFactory.h"
#include "itkCommand.h"
#include "vnl/vnl_vector.h"
#include "itkVectorImage.h"
#include "itkDefaultConvertPixelTraits.h"
#include "itkMetaImageIO.h"

namespace itk
{

//---------------------------------------------------------
template <class TInputImage>
ImageFileCastWriter<TInputImage>
::ImageFileCastWriter()
{
  this->m_Caster = 0;
  this->m_OutputComponentType = this->GetDefaultOutputComponentType();
}


//---------------------------------------------------------
template <class TInputImage>
std::string  
ImageFileCastWriter<TInputImage>
::GetDefaultOutputComponentType(void) const
{
  /** Make a dummy imageIO object, which has some handy functions */
  MetaImageIO::Pointer dummyImageIO = MetaImageIO::New();
  /** Set the pixeltype */
  typedef typename InputImageType::InternalPixelType ScalarType;
  dummyImageIO->SetPixelTypeInfo(typeid(ScalarType));  
  /** Get its description */
  return dummyImageIO->GetComponentTypeAsString(
    dummyImageIO->GetComponentType() );
}


//---------------------------------------------------------
template <class TInputImage>
ImageFileCastWriter<TInputImage>
::~ImageFileCastWriter()
{
  this->m_Caster = 0;
}


//---------------------------------------------------------
template <class TInputImage>
void 
ImageFileCastWriter<TInputImage>
::GenerateData(void)
{
  const InputImageType * input = this->GetInput();

  itkDebugMacro(<<"Writing file: " << this->GetFileName() );
  
  // Make sure that the image is the right type and no more than 
  // four components.
  typedef typename InputImageType::PixelType ScalarType;

  if( strcmp( input->GetNameOfClass(), "VectorImage" ) == 0 ) 
    {
    typedef typename InputImageType::InternalPixelType VectorImageScalarType;
    this->GetImageIO()->SetPixelTypeInfo( typeid(VectorImageScalarType) );
    
    typedef typename InputImageType::AccessorFunctorType AccessorFunctorType;
    this->GetImageIO()->SetNumberOfComponents( AccessorFunctorType::GetVectorLength(input) );
    }
  else
    {
    // Set the pixel and component type; the number of components.
    this->GetImageIO()->SetPixelTypeInfo(typeid(ScalarType));  
    }

  /** Setup the image IO for writing. */
  this->GetImageIO()->SetFileName( this->GetFileName() );
  
  /** Get the number of Components */
  unsigned int numberOfComponents = this->GetImageIO()->GetNumberOfComponents();
  
  /** Extract the data as a raw buffer pointer and possibly convert.
   * Converting is only possible if the number of components equals 1 */
  if ( 
    this->m_OutputComponentType != 
      this->GetImageIO()->GetComponentTypeAsString( this->GetImageIO()->GetComponentType() )
    && numberOfComponents == 1 )
  {
    void * convertedDataBuffer = 0;

    /** convert the scalar image to a scalar image with another componenttype 
     * The imageIO's PixelType is also changed */
    if ( this->m_OutputComponentType == "char" )
    {
      convertedDataBuffer = this->ConvertScalarImage<char>( input );
    }
    else if ( this->m_OutputComponentType == "unsigned_char" )
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned char>( input );
    }
    else if ( this->m_OutputComponentType == "short" )
    {
      convertedDataBuffer = this->ConvertScalarImage<short>( input );
    }
    else if ( this->m_OutputComponentType == "unsigned_short" )
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned short>( input );
    }
    else if ( this->m_OutputComponentType == "int" )
    {
      convertedDataBuffer = this->ConvertScalarImage<int>( input );
    }
    else if ( this->m_OutputComponentType == "unsigned_int" )
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned int>( input );
    }
    else if ( this->m_OutputComponentType == "long" )
    {
      convertedDataBuffer = this->ConvertScalarImage<long>( input );
    }
    else if ( this->m_OutputComponentType == "unsigned_long" )
    {
      convertedDataBuffer = this->ConvertScalarImage<unsigned long>( input );
    }
    else if ( this->m_OutputComponentType == "float" )
    {
      convertedDataBuffer = this->ConvertScalarImage<float>( input );
    }
    else if ( this->m_OutputComponentType == "double" )
    {
      convertedDataBuffer = this->ConvertScalarImage<double>( input );
    }
           
    /** Do the writing */
    this->GetImageIO()->Write( convertedDataBuffer );
    /** Release the caster's memory */
    this->m_Caster = 0;
    
  }
  else
  {
    /** No casting needed or possible, just write */
    const void* dataPtr = (const void*) input->GetBufferPointer();
    this->GetImageIO()->Write(dataPtr);
  }

}



} // end namespace itk

#endif
