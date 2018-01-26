NUM_DAT=887
for ((i=0; i<=$NUM_DAT; ++i))  
do  
printf -v file "%03d"  "${i}"
echo $file
./itkGPUResampleImageFilterTest -in /home/liuxinglong01/1HDD/data/LUNA/raw/$file.mhd -out /home/liuxinglong01/1HDD/data/LUNA/test/cpu.mhd /home/liuxinglong01/1HDD/data/LUNA/test/gpu.mhd -i "BSpline" -rmse 5
done 
