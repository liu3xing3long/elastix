import SimpleITK as sitk
import time
import os

test_type = "resample"

###################################################################
def test_resample():
    total_time = 0.0
    num_images = 10
    image_start = 1

    data_path = "/home/liuxinglong01/1HDD/data/LUNA/raw"
    output_path = "/home/liuxinglong01/1HDD/data/testdata"

    for ifile in range(image_start, image_start + num_images):
        filename = "{0}".format(ifile).zfill(3) + ".mhd"
        absname = os.path.join(data_path, filename)
        func = sitk.MedImageGPUFilter()
        img = sitk.ReadImage(absname)
        start = time.time()
        retImg = func.ResampleImage(img, [1.0, 1.0, 1.0])
        end = time.time()
        this_time = end - start
        total_time += this_time

        sitk.WriteImage(retImg, os.path.join(output_path, "{0}_gpu.mhd".format(ifile).zfill(3)))
        print "data {0}, time {1}".format(ifile, this_time)
    print "average time {0}".format(total_time / num_images)

###################################################################
def test_binary_dilate():
    data_path = "/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/test" 
    output_path = "/home/liuxinglong01/1HDD/work/vnet.pytorch/working_imgs_1mm_xyz/test" 
    
    data_img = "LKDS-00004_Threshold.mha"
    num_radiuses = 10
    radius_start = 1

    data_name, data_ext = os.path.split(data_img)
    img = sitk.ReadImage(os.path.join(data_path, data_img))
    func = sitk.MedImageGPUFilter()
    for radius in range(radius_start, radius_start + num_radiuses):
        start = time.time()
        retImg = func.BinaryDilateImage(img)
        end = time.time()
        this_time = end - start
        total_time += this_time
        sitk.WriteImage(retImg, os.path.join(output_path, "{0}_gpu_{1}_r{2}.mha".format(data_name, "dilate", radius)))
        print "raidus {0}, time {1}".format(raidus, this_time)
    print "average time {0}".format(total_time / num_radiuses)






if __name__=="__main__":
    if test_type == "resample":
        test_resample()
    elif test_type == "binarydilate":
        test_binary_dilate()
    else:
        print "do nothing"

