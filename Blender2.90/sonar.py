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

home_directory = os.getcwd()
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
# Set whether to produce hit, ele and fan-shaped images
HIT = False
ELE = False
FAN = True
RESCALE = False

# Sonar variables
uplimit = 3.336                                 # Sonar range, upper limit
lowlimit = 1.8                                  # Sonar range, lower limit
resolution = 0.003                              # Sonar range, resolution
print((uplimit - lowlimit) / resolution)
n_beams = 256                                   # Sonar beams

# Sonar imperfections
gs_kernel_size_b = 3                            # Gaussian blur, kernel size in beam direction
gs_kernel_size_r = 15                           # Gaussian blur, kernel size in range direction
gs_mean = 0                                     # Gaussian noise, mean
gs_var = 0.03                                   # Gaussian noise, variance
st_kernel_size = 5                              # Tangential streak, kernel (1 x size)
st_kernel_var = 11
x = np.linspace(
    -math.floor(st_kernel_size / 2),
    math.floor(st_kernel_size / 2),
    st_kernel_size
)
st_kernel = np.exp(-0.5 * (x / math.sqrt(st_kernel_var)) ** 2)
st_kernel = np.reshape(st_kernel / np.sum(st_kernel), (1, st_kernel_size))
# st_kernel = np.ones(1)
# st_kernel = np.ones((1, st_kernel_size), np.float32) / st_kernel_size
print(st_kernel)

# Camera variables
imagewidth = 0                                  # Blender camera, width (set below)
imageheight = 0                                 # Blender camera, height (set below)
focal_length = 0                                # Blender camera, focal length (set below)
horizontal_fov = 0                              # Blender camera, horizontal FOV (degs) (set below)
vertical_fov = 0                                # Blender camera, vertical FOV (degs) (set below)
scale = 1                                       # Downscaling (re: number of beams)
normalthres = 0.02
# max_integration = 3

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
    cv.imwrite('testcolor.png', r * 255)

    ################################################################################
    # depth with exr
    exrimage = OpenEXR.InputFile(fileNamedepth)

    dw = exrimage.header()["dataWindow"]
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    (rd, gd, bd) = [fromstr(s) for s in exrimage.channels("RGB")]

    def dist2depth_img(rd, focal=focal_length):
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

    arrd = np.array(rd)
    arrd = dist2depth_img(arrd)
    cv.imwrite('testdepth.png', arrd / uplimit * 255)

    # arr2 = arrd / uplimit
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

    (xn, yn, zn) = [fromstr(s) for s in exrimage.channels("RGB")]

    ######################################################################################
    # put img into array
    arrc = np.array(r)                              # Color (red)

    arrx = np.array(xn)
    arry = np.array(yn)
    arrz = np.array(zn)
    arrxyz = np.dstack((arrx, arry, arrz))
    cv.imwrite('testnormal.png', arrxyz * 255)      # Normal (xyz)
    arrxyz_diff = (
        np.abs(np.diff(arrx, axis=0, prepend=0)) +
        np.abs(np.diff(arry, axis=0, prepend=0)) +
        np.abs(np.diff(arrz, axis=0, prepend=0))
    )
    
    dmax = np.amax(arrd)
    dmin = np.amin(arrd)
    print("max distances:", dmax)
    print("min distances:", dmin)

    #################################################################################################################################
    ##################################################################################
    # resize image will lead to some sample problem, for high resolution image, this part is not needed
    # this part is difficult, to be improved
    rawc0 = arrc
    rawd0 = arrd

    #########################################################################################
    # cv.imshow("depth",rawd0/np.max(rawd0))
    # image generation
    ##here is the most important part, no research processed this part well, it is about how to add the intensities on the same arc
    ##change the mode to depth image generation or elevation angle image generation here

    # Calculate length of sonar image (from lowlimit to uplimit, divided to resolutions)
    length = int(np.floor((uplimit - lowlimit) / resolution))
    print(length)

    arrraw = np.zeros((length, imagewidth))
    ele = np.zeros((length, imagewidth))

    count = np.zeros((length, imagewidth), dtype=np.int32)
    # container = np.zeros((length, imagewidth), dtype=np.int32)
    
    # Conversion from measured depth to depth pixel (dp) on sonar image
    depth_pixels = np.floor((rawd0 - lowlimit) / resolution).astype(int)
    depth_pixels_diff = np.diff(depth_pixels, axis=0, prepend=0)
    current_depth = np.zeros(imagewidth, dtype=np.int32)

    # Loop through colors + depth + normal images
    mask = np.argwhere(
        (depth_pixels > 0) & (depth_pixels < length) &              # Check if within sonar range
        ((depth_pixels_diff != 0) | (arrxyz_diff > normalthres))    # Check if wasn't the same surface (previous dp not the same) or same but normal changed
    )
    # Add up all intensities at same depth
    for i in mask:
        arrraw[depth_pixels[tuple(i)], i[1]] += rawc0[tuple(i)]

    if ELE or HIT:
        for j in range(imagewidth):               # Vertical slices
            for i in range(imageheight):                # Horizontal slices
                dp = depth_pixels[i, j]
                if dp >= length or dp <= 0:
                    continue

                ## for the hitted points on the same arc. if the normal vector is larger than a threshold, then integrate the intensities
                # This is so that multiple pixels on depth image don't end up contributing to the same pixel on sonar image
                if (
                    depth_pixels_diff[i, j] == 0                            # Implying still in neighborhood of previous point
                    # This is unreliable for different angles
                    # and abs(i - container[dp, j]) < 4
                    and arrxyz_diff[i, j] < normalthres
                    
                    # This is too slow
                    # arrraw[dp, j] > 0                                       # Only does this skip if arrraw is already larger than 0
                    # and i + 1 < imageheight
                    # and abs(arrxyz[container[dp, j], j, 0] - arrxyz[i, j, 0]) < normalthres         # and current normal vector
                    # and abs(arrxyz[container[dp, j], j, 1] - arrxyz[i, j, 1]) < normalthres         # is very close to previous
                    # and abs(arrxyz[container[dp, j], j, 2] - arrxyz[i, j, 2]) < normalthres         # normal vector (or norm. vec. at i = 0)
                ):
                    # container=i
                    continue

                ## maximum integration: 3 times      
                # if count[dp, j] < 10:                                     # No maximum integration
                if True:
                    if ELE:
                        ele[dp, j] = i
                    
                    if HIT:
                        count[dp, j] = count[dp, j] + 1
                    # container[dp, j] = int(i)

                if dp != current_depth[j]:
                    current_depth[j] = dp

    hit = np.zeros((length, imagewidth))

    if HIT:
        for i in range(int(length)):
            for j in range(imagewidth):
                hit[i, j] = count[i, j]
                if hit[i, j] > 2:
                    hit[i, j] = 2

    # ==========================================================
    # Create sonar-accurate image (width: beams, height: depths)
    # Precompute map
    if not hasattr(imageGeneration, "beam_map"):
        # Polar coordinates centered at top left of image (beam 0 at FOV edge)
        tv, rv = np.meshgrid(
            np.linspace(-n_beams/2, n_beams/2 - 1, n_beams),
            np.linspace(0, length - 1, length)
        )

        # Create the maps
        imageGeneration.beam_height_map = (rv).astype(np.float32)
        imageGeneration.beam_width_map = (
            focal_length * np.tan(tv / n_beams * horizontal_fov * math.pi / 180) + imagewidth / 2
        ).astype(np.float32)
        imageGeneration.beam_map = True

    # Remap & flip image
    arrraw = np.flip(cv.remap(
        arrraw,
        imageGeneration.beam_width_map,
        imageGeneration.beam_height_map,
        cv.INTER_LANCZOS4
    ), axis=1)
    if ELE:
        np.flip(ele = cv.remap(
            ele,
            imageGeneration.beam_width_map,
            imageGeneration.beam_height_map,
            cv.INTER_LANCZOS4
        ), axis=1)
    if HIT:
        np.flip(hit = cv.remap(
            hit,
            imageGeneration.beam_width_map,
            imageGeneration.beam_height_map,
            cv.INTER_LANCZOS4
        ), axis=1)

    # ==========================================================
    # Add sonar imperfections
    arrraw = cv.GaussianBlur(arrraw, (gs_kernel_size_b, gs_kernel_size_r), 0)
    print(str(np.min(arrraw)) + ", " + str(np.max(arrraw)))
    arrraw += np.random.normal(gs_mean, gs_var ** 0.5, (arrraw.shape[0], arrraw.shape[1]))
    print(str(np.min(arrraw)) + ", " + str(np.max(arrraw)))
    arrraw = cv.filter2D(arrraw, -1, st_kernel)
    print(str(np.min(arrraw)) + ", " + str(np.max(arrraw)))
    cv.imwrite('testkernel.png', arrraw * 255)

    # ========================================================
    # smoothing, fill some holes due to aliasing
    # for i in range(int(length)):
    # for j in range(imagewidth):
    # if arrraw[i,j]<0.005 and i>2 and i+2 < int(length):
    # xm = 1
    # arrraw[i,j]=(arrraw[i-2,j]+arrraw[i-1,j]+arrraw[i+1,j]+arrraw[i+2,j])/4
    # hit[i,j]=(hit[i-2,j]+hit[i-1,j]+hit[i+1,j]+hit[i+2,j])/4.0
    # arrraw[i,j]=arrraw[i,j]
    # ========================================================
    length2 = int(math.ceil(uplimit / resolution))
    half_hfov_rad = horizontal_fov / 2 / 180 * math.pi
    width = int(math.ceil(uplimit / resolution * np.sin(half_hfov_rad))) * 2
    offset = int(math.ceil(length2 - length2 * math.cos(half_hfov_rad)))
    length3 = int(length2 + offset)
    fan = np.zeros((length3, width))
    if FAN:
        # To generate fan-shaped image
        # Precompute array
        if not hasattr(imageGeneration, "fan_map"):
            # Cartesian coordinates centered at top center of image
            xv, yv = np.meshgrid(
                np.linspace(-width/2, width/2 - 1, width),
                np.linspace(0, length2 - 1, length2)
            )
            # Polar coordinates centered at top center of image
            rv = np.sqrt(xv**2 + yv**2)
            tv = np.arctan2(xv, yv)

            # Limit visible range
            mask = (rv < uplimit / resolution) & (rv > lowlimit / resolution) & \
                (np.abs(tv) <= horizontal_fov * math.pi / 360)

            # Create the maps
            imageGeneration.fan_height_map = ((rv - lowlimit / resolution) * mask).astype(np.float32)
            imageGeneration.fan_width_map = ((tv / (horizontal_fov * math.pi / 180) + 0.5) * n_beams * mask).astype(np.float32)
            imageGeneration.fan_mask = mask
            imageGeneration.fan_map = True

        fan = np.flip(cv.remap(
            arrraw, 
            imageGeneration.fan_width_map,
            imageGeneration.fan_height_map,
            cv.INTER_LANCZOS4
        ) * imageGeneration.fan_mask)

        # print("width=", width)
        # print("length2=", length2)
        # # sample_interval=7.5/0.003*0.25/180*pi
        # # width=int(width/sample_interval)
        # # print('width=',width)
        # raw2fan0 = np.zeros((length2, width))
        # # raw2fan0=arrraw/np.amax(arrraw)
        # for i in range(length2):
        #     for j in range(width):
        #         root_squared = math.sqrt(i * i + j * j)
        #         idx = int(root_squared - lowlimit / resolution)
        #         if idx > 0:
        #             azi = int(
        #                 imagewidth * 180 / 32 / math.pi * math.acos(i / root_squared)
        #             )
        #             if idx > length - 1 or azi > imagewidth - 1:
        #                 continue
        #             raw2fan0[i, j] = arrraw[idx, azi]

        # # cv.imshow('fana',raw2fan0/np.amax(raw2fan0))
        # # save fanshaped image
        # # cv.imwrite('C:\\Users\\wang\\AppData\\Roaming\\Blender Foundation\\Blender\\2.80\\config\\BlenderPython\\result\\fan99.jpg', raw2fan0/np.amax(arrraw)*255)
        # #######################################################
        # # rotate fanshaped image

        # xcenter = 0
        # ycenter = int(length2)
        # for i in range(length3):
        #     for j in range(width):
        #         px = int(
        #             (j - xcenter) * math.cos(half_hfov_rad)
        #             + (i - ycenter) * math.sin(half_hfov_rad)
        #             + xcenter
        #         )
        #         py = int(
        #             -(j - xcenter) * math.sin(half_hfov_rad)
        #             + (i - ycenter) * math.cos(half_hfov_rad)
        #             + ycenter
        #         )
        #         if px < 0 or px > width - 1 or py > length2 - 1 or py < 0:
        #             continue
        #         if i - offset > 0:
        #             fan[i - offset, j] = raw2fan0[py, px]

        # # delete white edge
        # fan = fan[0 : length2 - 1, :]
        # fan = cv.flip(fan, 0)
        # fan = cv.flip(fan, 1)

    #################################################################################################################################
    # raw image
    # cv.imshow('rawsonar',arrraw/np.amax(arrraw))
    # save raw and resized sonar images
    common_id = str(num + 1).zfill(4) + ".png"
    savePath = os.path.join(save_directory, "rawimg_" + common_id)                          # Sonar image
    savePath1 = os.path.join(save_directory, "resizeimg_" + common_id)                      # Sonar image, resized
    savePath2 = os.path.join(save_directory, "fanshaped_" + common_id)                      # Sonar image, fan-shaped
    savePath3 = os.path.join(save_directory, "hit" + common_id)                             # Hits (if rays hit objects)
    savePath4 = os.path.join(save_directory, "hit_resize" + common_id)                      # Hits, resized
    savePath5 = os.path.join(save_directory, "ele_" + common_id)                            # Elevation (probably for ML dataset generation)
    savePath6 = os.path.join(save_directory, "ele_resize_" + common_id)                     # Elevation, resized

    # Sonar
    raw = arrraw / 1.0 * 255
    cv.imwrite(savePath, raw)  # np.amax(arrraw)  0.5 #0.7
    if RESCALE:
        rescale_raw = cv.resize(
            raw,
            dsize=(int(imagewidth / scale), int(length)),
            interpolation=cv.INTER_CUBIC,
        )
        cv.imwrite(savePath1, rescale_raw)
    # print(np.amax(arrraw))
    # print(np.amax(hit))

    if FAN:
        cv.imwrite(savePath2, fan / 1.0 * 255)  # 0.7

    # Hits
    if HIT:
        hit_raw = hit / 2 * 255
        cv.imwrite(savePath3, hit_raw)
        if RESCALE:
            hit_rescale_raw = cv.resize(
                hit_raw,
                dsize=(int(imagewidth / scale), int(length)),
                interpolation=cv.INTER_CUBIC,
            )
            cv.imwrite(savePath4, hit_rescale_raw)

    # Elevation
    if ELE:
        ele_raw = ele / 560 * 255 #?
        cv.imwrite(savePath5, ele_raw)
        if RESCALE:
            ele_rescale_raw = cv.resize(
                ele_raw,
                dsize=(int(imagewidth / scale), int(length)),
                interpolation=cv.INTER_CUBIC,
            )
            cv.imwrite(savePath6, ele_rescale_raw)

    return [arrraw, hit / 2, ele / 560, fan]


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

    spot_light = bpy.data.objects["Spot.000"]
    spot_light.data.energy = 1.0

    # bpy.context.scene.camera = bpy.context.scene.objects["Camera.002"]
    # sceneRender()
    # opticalimageGeneration(i)

    # cam = bpy.data.objects['Camera.002']
    # print(get_calibration_matrix_K_from_blender(cam.data))

    # generate sonar image according to current camera perspective
    cam = bpy.data.objects["Camera"]
    K = get_calibration_matrix_K_from_blender(cam.data)
    imagewidth = int(K[0][2] * 2)
    imageheight = int(K[1][2] * 2)
    focal_length = K[0][0]
    horizontal_fov = math.atan2(K[0][2], K[0][0]) * 180 / math.pi * 2
    vertical_fov = math.atan2(K[1][2], K[0][0]) * 180 / math.pi * 2

    bpy.context.scene.camera = bpy.context.scene.objects["Camera"]
    sceneRender()
    raw, hit_raw, ele_raw, fan = imageGeneration(i, 0)
    
    line = str(i) + "\n"
    f.write(line)


f.close()
time_end = time.time()
print("Total cost", time_end - time_start)
