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

//
// Brute-force mean filter
//
// The current implementation does not use shared memory, so this could be
// very slow on older GPUs (pre-Fermi) that do not have hardware cache.
//

void quick_sort(float* nums, int b, int e)
{
    float tmp;
    if (b < e - 1) 
    {
        int lb = b, rb = e - 1;
        while (lb < rb) 
        {
            while (nums[rb] >= nums[b] && lb < rb)
                rb--;
            while (nums[lb] <= nums[b] && lb < rb)
                lb++;
            // quick_sort_swap(nums[lb], nums[rb]);
            tmp = nums[lb];
            nums[lb] = nums[rb];
            nums[rb] = tmp;
        }
        // quick_sort_swap(nums[b], nums[lb]);
        tmp = nums[b];
        nums[b] = nums[lb];
        nums[lb] = tmp;

        quick_sort(nums, b, lb);
        quick_sort(nums, lb + 1, e);
    }
}

void selection_sort(float* nums, unsigned int start, unsigned int end)
{
    float temp;
    int i, j, min;
    for (i = start; i < end; i++) 
    {
        min = i;
        for (j = i + 1; j < end; j++) 
        {
            if (nums[j] < nums[min]) 
            {
                min = j;
            }
        }

        temp = nums[i];
        nums[i] = nums[min];
        nums[min] = temp;
    }
}

#ifdef DIM_1
#define MAX_RADIUS 1024
#define MAX_D (MAX_RADIUS * 2 + 1)

__kernel void MedianFilter(__global const PIXELTYPE* in,__global PIXELTYPE* out, int radiusx, int width)
{
  int gix = get_global_id(0);
  // float sum = 0;
  unsigned int num = 0;
  if(gix < width)
  {
    // int buffer[MAX_RADIUS];
  
    // Zero-flux boundary condition
    num = 2*radiusx + 1;
    PIXELTYPE buffer[MAX_D];
    unsigned int buffer_idx = 0;

    for(int x = gix-radiusx; x <= gix+radiusx; x++)
    {
      unsigned int cidx = (unsigned int)(min(max(0, x),width-1));
      buffer[buffer_idx++] = in[cidx];
    }

    quickSort(buffer, 0, buffer_idx);
    out[gidx] = (PIXELTYPE)(buffer[buffer_idx / 2]);
  }
}
#endif

#ifdef DIM_2
#define MAX_RADIUS 64
#define MAX_D (MAX_RADIUS * 2 + 1)

__kernel void MedianFilter(__global const PIXELTYPE* in,
                         __global PIXELTYPE* out,
                         int radiusx, int radiusy, int width, int height)
{
  unsigned int MAX_RADIUS = 512;

  int gix = get_global_id(0);
  int giy = get_global_id(1);
  unsigned int gidx = width*giy + gix;
  // float sum = 0;
  unsigned int   num = 0;

  if(gix < width && giy < height)
  {
    // int buffer[MAX_RADIUS];
  
    // Zero-flux boundary condition
    num = (2*radiusx + 1)*(2*radiusy + 1);
    PIXELTYPE buffer[MAX_D * MAX_D];
    unsigned int buffer_idx = 0;

    for(int y = giy-radiusy; y <= giy+radiusy; y++)
    {
      unsigned int yid = (unsigned int)(min(max(0, y),height-1));
      for(int x = gix-radiusx; x <= gix+radiusx; x++)
      {
        unsigned int cidx = width*yid + (unsigned int)(min(max(0, x),width-1));
        buffer[buffer_idx++] = in[cidx];
      }
    }

    quickSort(buffer, 0, buffer_idx);
    out[gidx] = (PIXELTYPE)(buffer[buffer_idx / 2]);
  }
}
#endif

#ifdef DIM_3
#define MAX_RADIUS 8
#define MAX_D (MAX_RADIUS * 2 + 1)

__kernel void MedianFilter(const __global PIXELTYPE* in,
                         __global PIXELTYPE* out,
                         int radiusx, int radiusy, int radiusz,
                         int width, int height, int depth)
{

  int gix = (int)get_global_id(0);
  int giy = (int)get_global_id(1);
  int giz = (int)get_global_id(2);

  unsigned int gidx = width*(giz*height + giy) + gix;

  // float sum = 0;
  unsigned int num = 0;

  /* NOTE: More than three-level nested conditional statements (e.g.,
     if A && B && C..) invalidates command queue during kernel
     execution on Apple OpenCL 1.0 (such Macbook Pro with NVIDIA 9600M
     GT). Therefore, we flattened conditional statements. */
  bool isValid = true;
  if(gix >= width) isValid = false;
  if(giy >= height) isValid = false;
  if(giz >= depth) isValid = false;

  if( isValid )
  {
    // Zero-flux boundary condition
    num = (2*radiusx + 1)*(2*radiusy + 1)*(2*radiusz + 1);
    
    float buffer[MAX_D * MAX_D * MAX_D];
    unsigned int buffer_idx = 0;

    for(int z = giz-radiusz; z <= giz+radiusz; z++)
    {
      unsigned int zid = (unsigned int)(min(max(0, z),depth-1));
      for(int y = giy-radiusy; y <= giy+radiusy; y++)
      {
        unsigned int yid = (unsigned int)(min(max(0, y),height-1));
        for(int x = gix-radiusx; x <= gix+radiusx; x++)
        {
          unsigned int cidx = width*(zid*height + yid) + (unsigned int)(min(max(0, x),width-1));
          buffer[buffer_idx++] = ( float )(in[cidx]);
        }
      }
    }

    // selection_sort(buffer, 0, buffer_idx);
    // selection sort
    float temp;
    for (unsigned int i = 0; i < buffer_idx; i++) 
    {
        unsigned int min = i;
        for (unsigned int j = i + 1; j < buffer_idx; j++) 
            if (buffer[j] < buffer[min]) 
                min = j;
        temp = buffer[i];
        buffer[i] = buffer[min];
        buffer[min] = temp;
    }

    unsigned int target_index = (unsigned int)(buffer_idx / 2);
    out[gidx] = (PIXELTYPE)(buffer[target_index]);
    // out[gidx] = (PIXELTYPE)(buffer_idx);
  }

}
#endif
