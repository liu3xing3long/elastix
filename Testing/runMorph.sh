# !/usr/bin/bash

NUM_DAT=887
for ((i=0; i<=$NUM_DAT; ++i))  
do  
printf -v file "%03d"  "${i}"
echo $file
echo "dialting"

./itkGPUBinaryDilateImageFilterTest3  "/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/test/LKDS-00004_Threshold.mha" "/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/test/LKDS-00004_Threshold_gpu_dilate.mha" 1 0 0 5

echo "erodeing"

./itkGPUBinaryDilateImageFilterTest3  "/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/test/LKDS-00004_Threshold.mha" "/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/test/LKDS-00004_Threshold_gpu_dilate.mha" 1 0 0 5

# done 
