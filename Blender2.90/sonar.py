"""
For Blender 2.9
Written by Yusheng Wang, wang@robot.t.u-tokyo.ac.jp, 2021.4.30
"""
# import libraries
import os
import bpy
import numpy as np
import sys
import OpenEXR
import math
import mathutils
import cv2 as cv
import time
from mathutils import Matrix
from mathutils import Vector

home_directory = "E:\\NYU\\ENGR\\ENGR-UH 4011\\Sonar-simulator-blender\\Blender2.90\\"
save_directory = os.path.join(home_directory, "output")
bpy.ops.wm.open_mainfile(
    filepath=os.path.join(home_directory, "acousticimagegeneration_z.blend")
)

print(
    "********************************************************************************************"
)
##change here!!!
###some important parameters here
## set the uplimit and lowlimit of the image i.e. sensing range
# GLOBAL VARIABLES
uplimit = 3.336  # 4.336
lowlimit = 1.8  # 4.336
resolution = 0.003
imagewidth = int(1280)
imageheight = int(560)
scale = 1  # 10
scaledimagewidth = int(imagewidth / scale)
normalthres = 0.02
max_integration = 3

#######################################################################################
# count time start
time_start = time.time()


def sceneRender():
    # generate 1. depth image  2. grayscale image 3. normal map in world coordinate
    # Use node

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # create input render layer node
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = 185, 285  # what does this 'location' mean

    # create output node
    v = tree.nodes.new("CompositorNodeOutputFile")
    v.location = 750, 210
    # v.use_alpha = False
    v.base_path = os.path.join(home_directory, "depth")

    o = tree.nodes.new("CompositorNodeOutputFile")
    o.location = 750, 320
    o.base_path = os.path.join(home_directory, "img")

    n = tree.nodes.new("CompositorNodeOutputFile")
    n.location = 750, 440
    n.base_path = os.path.join(home_directory, "normal")

    # Links
    links.new(rl.outputs[2], v.inputs[0])  # link Image output to Viewer input
    links.new(rl.outputs[0], o.inputs[0])
    links.new(rl.outputs[4], n.inputs[0])

    # render
    bpy.ops.render.render()

    # check how long it takes for saving and render
    # time_end=time.time()
    # print('render cost',time_end-time_start)


################################################################################
# print("image generated")
def opticalimageGeneration(num):
    # opencv read
    fileNamergb = os.path.join(home_directory, "img", "image0038.exr")
    fileNamedepth = os.path.join(home_directory, "depth", "image0038.exr")
    fileNamenormal = os.path.join(home_directory, "normal", "image0038.exr")
    # im = cv.imread(fileName,0)
    # cv.imshow('test', im)

    #################################################################################
    # rgb with exr
    exrimage = OpenEXR.InputFile(fileNamergb)

    dw = exrimage.header()["dataWindow"]
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    def fromstr(s):
        mat = np.fromstring(s, dtype=np.float32)
        mat = mat.reshape(height, width)
        return mat

    (r, g, b) = [fromstr(s) for s in exrimage.channels("RGB")]
    img = r
    arr = np.array(img)
    # print(arr)
    arr = arr / np.amax(arr)
    # cv.resizeWindow('test',128,56)
    # cv.imshow('testgrayscale', arr)
    rr = np.array(r)
    gg = np.array(g)
    bb = np.array(b)
    rgb = np.zeros([height, width, 3], dtype=np.float32)
    rgb[:, :, 0] = bb
    rgb[:, :, 1] = gg
    rgb[:, :, 2] = rr

    ################################################################################
    # depth with exr
    exrimage = OpenEXR.InputFile(fileNamedepth)

    dw = exrimage.header()["dataWindow"]
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    (rd, gd, bd) = [fromstr(s) for s in exrimage.channels("RGB")]
    img2 = rd
    # print(img2)
    arr2 = np.array(img2)

    arr2 = arr2 / uplimit
    # cv.resizeWindow('test2',128,56)
    # cv.imshow('testdepth', arr2)
    ##test
    # print(arr2[554,2])
    # print(arr2[554,640])

    ##################################################################################
    # normal with exr
    exrimage = OpenEXR.InputFile(fileNamenormal)

    dw = exrimage.header()["dataWindow"]
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    (rn, gn, bn) = [fromstr(s) for s in exrimage.channels("RGB")]

    imgrgb = np.zeros([height, width, 3], dtype=np.float32)

    img3 = bn
    arr3 = np.array(img3)
    arr3 = arr3 / np.max(arr3)
    img4 = gn
    arr4 = np.array(img4)
    arr4 = arr4 / np.max(arr4)
    img5 = rn
    arr5 = np.array(img5)
    arr5 = arr5 / np.max(arr5)
    imgrgb[:, :, 0] = arr3
    imgrgb[:, :, 1] = arr4
    imgrgb[:, :, 2] = arr5
    # cv.imshow('testrgb', imgrgb)

    print("image read")
    print("width", width)
    print("height", height)

    ######################################################################################
    # put img into array
    arrc = np.array(img)
    arrd = np.array(img2)
    arrb = np.array(img3)
    arrg = np.array(img4)
    arrr = np.array(img5)
    arrrgb = np.zeros([height, width, 3], dtype=np.float32)
    arrrgb[:, :, 0] = arrb
    arrrgb[:, :, 1] = arrg
    arrrgb[:, :, 2] = arrr

    path = os.path.join(home_directory, "opti_front" + str(num + 1) + ".png")
    cv.imwrite(path, 1.0 / arrd * 255)

    path = os.path.join(home_directory, "opti_if" + str(num + 1) + ".png")
    cv.imwrite(path, rgb * 255)

    path = os.path.join(home_directory, "opti_sfront" + str(num + 1) + ".txt")
    np.savetxt(path, 1.0 / arrd)

    # path = "D:\\self-a2fnet\\color_test\\opti_sif"+str(num+1)+".txt"
    # np.savetxt(path,arrc)


# @jit(nopython=True)
def imageGeneration(num, k):  # num:number of iteration -> number of image
    # opencv read
    fileNamergb = os.path.join(home_directory, "img", "image0038.exr")
    fileNamedepth = os.path.join(home_directory, "depth", "image0038.exr")
    fileNamenormal = os.path.join(home_directory, "normal", "image0038.exr")
    # im = cv.imread(fileName,0)
    # cv.imshow('test', im)

    #################################################################################
    # rgb with exr
    exrimage = OpenEXR.InputFile(fileNamergb)

    dw = exrimage.header()["dataWindow"]
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    def fromstr(s):
        mat = np.fromstring(s, dtype=np.float32)
        mat = mat.reshape(height, width)
        return mat

    (r, g, b) = [fromstr(s) for s in exrimage.channels("RGB")]
    img = r
    arr = np.array(img)
    # print(arr)
    arr = arr / np.amax(arr)
    # cv.resizeWindow('test',128,56)
    # cv.imshow('testgrayscale', arr)
    rr = np.array(r)
    gg = np.array(g)
    bb = np.array(b)
    rgb = np.zeros([height, width, 3], dtype=np.float32)
    rgb[:, :, 0] = bb
    rgb[:, :, 1] = gg
    rgb[:, :, 2] = rr

    ################################################################################
    # depth with exr
    exrimage = OpenEXR.InputFile(fileNamedepth)

    dw = exrimage.header()["dataWindow"]
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    (rd, gd, bd) = [fromstr(s) for s in exrimage.channels("RGB")]

    def dist2depth_img(rd, focal=2280.4170):
        img_width = rd.shape[1]
        img_height = rd.shape[0]

        # Get x_i and y_i (distances from optical center)
        cx = img_width // 2
        cy = img_height // 2

        xs = np.arange(img_width) - cx
        ys = np.arange(img_height) - cy
        xis, yis = np.meshgrid(xs, ys)

        depth = np.sqrt((1 + (xis**2 + yis**2) / (focal**2)) * rd**2)

        return depth

    img2 = rd
    arrd = np.array(img2)
    # print(img2)
    arr2 = np.array(img2)

    arrd = dist2depth_img(arrd)
    arr2 = dist2depth_img(arr2)

    arr2 = arr2 / uplimit
    # cv.resizeWindow('test2',128,56)
    # cv.imshow('testdepth', arr2)
    ##test
    # print(arr2[554,2])
    # print(arr2[554,640])

    ##################################################################################
    # normal with exr
    exrimage = OpenEXR.InputFile(fileNamenormal)

    dw = exrimage.header()["dataWindow"]
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    (rn, gn, bn) = [fromstr(s) for s in exrimage.channels("RGB")]

    imgrgb = np.zeros([height, width, 3], dtype=np.float32)

    img3 = bn
    arr3 = np.array(img3)
    arr3 = arr3 / np.max(arr3)
    img4 = gn
    arr4 = np.array(img4)
    arr4 = arr4 / np.max(arr4)
    img5 = rn
    arr5 = np.array(img5)
    arr5 = arr5 / np.max(arr5)
    imgrgb[:, :, 0] = arr3
    imgrgb[:, :, 1] = arr4
    imgrgb[:, :, 2] = arr5
    # cv.imshow('testrgb', arr3)

    print("image read")
    print("width", width)
    print("height", height)

    ######################################################################################
    # put img into array
    arrc = np.array(img)

    arrb = np.array(img3)
    arrg = np.array(img4)
    arrr = np.array(img5)
    arrrgb = np.zeros([height, width, 3], dtype=np.float32)
    arrrgb[:, :, 0] = arrb
    arrrgb[:, :, 1] = arrg
    arrrgb[:, :, 2] = arrr
    # print(len(rd))
    # print(arrd)
    dmax = np.amax(arrd)
    dmin = np.amin(arrd)
    print("max distances:", dmax)
    print("min distances:", dmin)
    length = np.floor((uplimit - lowlimit) / resolution)
    length = int(length)
    print(length)

    #################################################################################################################################
    ##################################################################################
    # resize image will lead to some sample problem, for high resolution image, this part is not needed
    # this part is difficult, to be improved
    print("resize image")
    rawd0 = np.zeros((imageheight * imagewidth))
    rawd0 = rawd0.reshape(imageheight, imagewidth)
    rawc0 = np.zeros((imageheight * imagewidth))
    rawc0 = rawc0.reshape(imageheight, imagewidth)
    for i in range(imageheight):
        for j in range(scaledimagewidth):
            j0 = j * scale
            rawd0[i, j] = arrd[i, j0]
            rawc0[i, j] = arrc[i, j0]

    #########################################################################################
    # cv.imshow("depth",rawd0/np.max(rawd0))
    # image generation
    ##here is the most important part, no research processed this part well, it is about how to add the intensities on the same arc
    ##change the mode to depth image generation or elevation angle image generation here

    # arrraw=np.zeros(128*length,dtype=np.float32)
    # arrraw=arrraw.reshape(length,128)
    arrraw = np.zeros((length, scaledimagewidth))
    ele = np.zeros((length, scaledimagewidth))

    arrtensor = np.zeros((imageheight, length, scaledimagewidth))
    rawd1 = rawd0
    count = np.zeros((length, scaledimagewidth))
    container = np.zeros((length, scaledimagewidth), dtype=np.int32)
    intensitymax = np.amax(rawc0)
    intensitymin = np.amin(rawc0)
    para = 1 / intensitymax
    # print(rawc0)
    # print(para)
    # add intensity
    # container=-1
    print(scaledimagewidth)
    print(imageheight)
    for j in range(scaledimagewidth):
        for i in range(imageheight):
            dp0 = np.floor((rawd0[i, j] - lowlimit) / resolution)
            dp = int(dp0)
            ## maximum integration: 3 times

            if dp >= length or dp <= 0:
                continue

            ## for the hitted points on the same arc. if the normal vector is larger than a threshold, then integrate the intensities
            if (
                arrraw[dp, j] > 0
                and i + 1 < imageheight
                and abs(arrrgb[container[dp, j], j, 0] - arrrgb[i, j, 0]) < normalthres
                and abs(arrrgb[container[dp, j], j, 1] - arrrgb[i, j, 1]) < normalthres
                and abs(arrrgb[container[dp, j], j, 2] - arrrgb[i, j, 2]) < normalthres
            ):
                # container=i
                continue
            ## maximum integration: 3 times
            if count[dp, j] < 2:
                # if i==container[dp,j]+1 or i==container[dp,j]+2:
                #    continue

                # if arrraw[dp,j]==0:
                # arrraw[dp,j]=i
                # arrraw[dp,j]=dp*math.sin((-14+i/560*28)/180*math.pi)+length/2    # change the intensity to depth
                # if arrraw[dp,j]>0:
                #    if rawc0[i,j]<0.005:
                #        continue
                arrtensor[i, dp, j] = 1
                # add the intensity
                # if arrraw[dp,j]==0:
                # ele1[dp,j]=i
                # else:
                # ele2[dp,j]=i
                ele[dp, j] = i
                arrraw[dp, j] = rawc0[i, j] + arrraw[dp, j]
                # arrraw[dp,j]=count[dp,j] #change the gray scale intensity to hit times
                # arrraw[dp,j]=i # change the intensity to elevation angle
                # arrraw[dp,j]=arrraw[dp,j]+1/(1+np.exp(-(rawc0[i,j])*para-0.5)*10)
                # print(arrraw[dp,j])
                # arrraw[dp,j]=rawc0[i,j]
                count[dp, j] = count[dp, j] + 1
                container[dp, j] = int(i)
            # print(arrrgb[container[dp,j], j, 0])

    # writePath = "D:\\elevate-estimation\\rec-ele\\rec"+str(num+1)+".txt"

    hit = np.zeros((length, scaledimagewidth))

    for i in range(int(length)):
        for j in range(scaledimagewidth):
            hit[i, j] = count[i, j]
            if hit[i, j] == 4 or hit[i, j] == 3:
                hit[i, j] = 2
                print("wtf")
            # if count[i,j]>1:
            # arrraw[i,j]=arrraw[i,j]/1    #count[i,j]
            # arrraw[i,j]=1+1/(1+np.exp(-(arrraw[i,j])*para-0.5)*10)
            # arrraw[i,j]=arrraw[i,j]/1
            # print(arrraw[i,j])
            # print(count[i,j])

    # smoothing, fill some holes due to aliasing
    # for i in range(int(length)):
    # for j in range(scaledimagewidth):
    # if arrraw[i,j]<0.005 and i>2 and i+2 < int(length):
    # xm = 1
    # arrraw[i,j]=(arrraw[i-2,j]+arrraw[i-1,j]+arrraw[i+1,j]+arrraw[i+2,j])/4
    # hit[i,j]=(hit[i-2,j]+hit[i-1,j]+hit[i+1,j]+hit[i+2,j])/4.0
    # arrraw[i,j]=arrraw[i,j]
    # ========================================================
    # To generate fan-shaped image
    width = int(uplimit / resolution * np.sin(32 / 180 * math.pi))
    print("width=", width)
    length2 = int(uplimit / resolution)
    print("length2=", length2)
    # sample_interval=7.5/0.003*0.25/180*pi
    # width=int(width/sample_interval)
    # print('width=',width)
    raw2fan0 = np.zeros((length2, width))
    # raw2fan0=arrraw/np.amax(arrraw)
    for i in range(length2):
        for j in range(width):
            root_squared = math.sqrt(i * i + j * j)
            idx = int(root_squared - lowlimit / resolution)
            if idx > 0:
                azi = int(
                    scaledimagewidth * 180 / 32 / math.pi * math.acos(i / root_squared)
                )
                if idx > length - 1 or azi > scaledimagewidth - 1:
                    continue
                raw2fan0[i, j] = arrraw[idx, azi]

    # cv.imshow('fana',raw2fan0/np.amax(raw2fan0))
    # save fanshaped image
    # cv.imwrite('C:\\Users\\wang\\AppData\\Roaming\\Blender Foundation\\Blender\\2.80\\config\\BlenderPython\\result\\fan99.jpg', raw2fan0/np.amax(arrraw)*255)
    #######################################################
    # rotate fanshaped image

    width2 = int(length2 * math.cos(74 / 180 * math.pi) * 2)
    offset = int(length2 - length2 * math.sin(74 / 180 * math.pi))
    length3 = int(length2 + offset)
    xcenter = 0
    ycenter = int(length2)
    degrees = 16
    halfwidth2 = int(width2 / 2)
    fan1 = np.zeros((length3, width2))
    for i in range(length3):
        for j in range(width2):
            px = int(
                (j - xcenter) * math.cos(degrees / 180 * math.pi)
                + (i - ycenter) * math.sin(degrees / 180 * math.pi)
                + xcenter
            )
            py = int(
                -(j - xcenter) * math.sin(degrees / 180 * math.pi)
                + (i - ycenter) * math.cos(degrees / 180 * math.pi)
                + ycenter
            )
            if px < 0 or px > width - 1 or py > length2 - 1 or py < 0:
                continue
            if i - offset > 0:
                fan1[i - offset, j] = raw2fan0[py, px]

    # delete white edge
    fan1 = fan1[0 : length2 - 1, :]
    fan1 = cv.flip(fan1, 0)
    fan1 = cv.flip(fan1, 1)
    #################################################################################################################################
    # raw image
    # cv.imshow('rawsonar',arrraw/np.amax(arrraw))
    # save raw and resized sonar images
    savePath = os.path.join(save_directory, "rawimg_" + str(num + 1) + "_" + str(k) + ".png")           # Sonar image
    savePath1 = os.path.join(save_directory, "resizeimg_" + str(num + 1) + "_" + str(k) + ".png")       # Sonar image, resized
    savePath2 = os.path.join(save_directory, "fanshaped_" + str(num + 1) + "_" + str(k) + ".png")       # Sonar image, fan-shaped
    savePath3 = os.path.join(save_directory, "hit" + str(num + 1) + ".jpg")                             # Hits (if rays hit objects)
    savePath4 = os.path.join(save_directory, "hit_resize" + str(num + 1) + ".jpg")                      # Hits, resized
    savePath5 = os.path.join(save_directory, "ele_" + str(num + 1) + "_" + str(k) + ".png")             # Elevation angle (probably for ML dataset generation)
    savePath6 = os.path.join(save_directory, "ele_resize_" + str(num + 1) + "_" + str(k) + ".png")      # Elevation angle, resized

    cv.imwrite(savePath, arrraw / 1.0 * 255)  # np.amax(arrraw)  0.5 #0.7
    print(np.amax(arrraw))
    print(np.amax(hit))
    cv.imwrite(savePath3, hit / 2 * 255)

    cv.imwrite(savePath5, ele / 560 * 255)

    # rescale raw image
    raw = cv.imread(savePath, 0)
    rescale_raw = cv.resize(
        raw,
        dsize=(int(scaledimagewidth / 10), int(length)),
        interpolation=cv.INTER_CUBIC,
    )
    cv.imwrite(savePath1, rescale_raw)

    hit_raw = cv.imread(savePath3, 0)
    hit_rescale_raw = cv.resize(
        hit_raw,
        dsize=(int(scaledimagewidth / 10), int(length)),
        interpolation=cv.INTER_CUBIC,
    )
    cv.imwrite(savePath4, hit_rescale_raw)

    ele1_raw = cv.imread(savePath5, 0)
    ele1_rescale_raw = cv.resize(
        ele1_raw,
        dsize=(int(scaledimagewidth / 10), int(length)),
        interpolation=cv.INTER_CUBIC,
    )
    cv.imwrite(savePath6, ele1_rescale_raw)

    # path = os.path.join(home_directory, "acoustic_if_"+str(num+1)+"_"+str(k)+".png")
    # cv.imwrite(path, rgb/1.0*255) #0.7

    cv.imwrite(savePath2, fan1 / 1.0 * 255)  # 0.7


def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
    return K


def clockwise(camera):
    camera.rotation_euler[2] = camera.rotation_euler[2] + 3 / 180 * 3.14159


def unclockwise(camera):
    camera.rotation_euler[2] = camera.rotation_euler[2] - 3 / 180 * 3.14159


def move(camera):
    camera.location.x = camera.location.x - 0.1

np.random.seed(0)
flag = 0
motion_file = "motion.txt"
f = open(os.path.join(save_directory, motion_file), mode="w")

camera = bpy.data.objects["Camera"]
area_light = bpy.data.objects["Area"]
area_light.data.energy = 0.0

for i in range(1):  # loop for generating images
    print("start!!!!!!!!!!!!!!!!!!!!!!!")

    camera.location.x = 0.8  # 2.3
    camera.location.y = 0.0
    camera.location.z = 1.5  # 2.1 2.5

    camera.rotation_euler[0] = 51 / 180 * 3.14159  # 51
    camera.rotation_euler[1] = 0
    camera.rotation_euler[2] = 90 / 180 * 3.14159

    camera.location.x = 0.8 - np.random.rand() * 2.0
    camera.location.y = 0 + (np.random.rand() * 2 - 1) * 4.0

    spot_light = bpy.data.objects["Spot.000"]
    spot_light.data.energy = 1.0

    # bpy.context.scene.camera = bpy.context.scene.objects["Camera.002"]
    # sceneRender()
    # opticalimageGeneration(i)

    # cam = bpy.data.objects['Camera.002']
    # print(get_calibration_matrix_K_from_blender(cam.data))

    # generate sonar image according to current camera perspective

    cam = bpy.data.objects["Camera"]
    print(get_calibration_matrix_K_from_blender(cam.data))

    bpy.context.scene.camera = bpy.context.scene.objects["Camera"]
    sceneRender()
    imageGeneration(i, 0)
    
    line = str(i) + "\n"
    f.write(line)


f.close()
time_end = time.time()
print("Total cost", time_end - time_start)
