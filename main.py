import os
import time
import argparse
import numpy as np
import matplotlib.image as ims
import matplotlib.pyplot as plt
import skimage as sk
from skimage import color
from scipy import ndimage
from os import listdir
from os.path import isfile, join
# from PIL import Image

### PARSER ###

parser = argparse.ArgumentParser(description='Automatic Box Measurement.')
parser.add_argument('-a', '--action', default="find_dim")
parser.add_argument('-y', '--ylen', type=float)
parser.add_argument('-x', '--xlen', type=float)
parser.add_argument('-i', '--input', default='test1.jpg')
parser.add_argument('-d', '--disp',  action='store_true')
parser.add_argument('-v', '--verbose',  action='store_true')
args = parser.parse_args()

### END PARSER ###

### FILES SETUP ###

CUR_DIR, OUT_DIR, CAL_DIR = "in/", "out/", "cal/"
cal_text_name = "cal.txt"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
if not os.path.exists(CAL_DIR):
    os.makedirs(CAL_DIR)

### END FILES SETUP ###

SMALL_THRESH = 500 #cm^3
MEDIUM_THRESH = 5000 #cm^3

### HELPER FUNCTIONS ###

# print only if verbose setting ON
def vprint(s, flag=args.verbose):
    if flag:
        print s

# display image to screen for viewing
def disp_image(imgs, pts=[], flag=args.disp, cmap=None):
    if flag:
        fig = plt.figure()
        for i in range(len(imgs)):
            fig.add_subplot(1, len(imgs), i+1)
            plt.imshow(imgs[i], cmap=cmap)
            if len(pts) != 0:
                if pts[i] != None:
                    plt.plot(pts[i][:,0], pts[i][:,1], 'o')
        plt.show()

# load an image using specified file_name
def load_image(file_name, direc=CUR_DIR):
    img = plt.imread(direc + str(file_name))
    height_file = open(direc + file_name[:-4] + ".txt", 'r')
    height = float(height_file.readline())
    height_file.close()
    # NOTE: Use in case opening image files becomes an issue
    # try:
    #     img = plt.imread(direc + str(file_name))
    # except IOError:
    #     img = Image.open(direc + str(file_name))
    #     try:
    #         img = np.array(img.getdata()).reshape(480, 640, 3)
    #     except IOError:
    #         img = np.array(img.getdata()).reshape(480, 640, 3)
    return img, height

### END HELPER FUNCTIONS ###

# get gradient matrix using input values along x and y axes
def get_grad(img, y_vals, x_vals):
    grad = np.zeros(img.shape)
    for x in range(img.shape[1]):
        grad[:, x, 0] += (y_vals*1. / np.max(y_vals))
        grad[:, x, 1] += (y_vals*1. / np.max(y_vals))
        grad[:, x, 2] += (y_vals*1. / np.max(y_vals))

    for x in range(img.shape[0]):
        grad[x, :, 0] += (x_vals*1. / np.max(x_vals))
        grad[x, :, 1] += (x_vals*1. / np.max(x_vals))
        grad[x, :, 2] += (x_vals*1. / np.max(x_vals))

    return grad

# convert pixels to centimeters
def calibrate(cal_dir=CAL_DIR):
    cf = open(cal_dir+cal_text_name, 'r')
    for line in cf:
        if "DIMENSIONS" in line:
            dim = [float(x) for x in line[line.index(':')+1:-1].split(',')[:2]]
    cf.close()

    files = [f for f in listdir(cal_dir) if isfile(join(cal_dir, f)) and '.jpg' in f ]
    h_files = [f for f in listdir(cal_dir) if isfile(join(cal_dir, f)) and '.txt' in f ]
    h_files.remove(cal_text_name)
    heights = [float(open(cal_dir+f, 'r').readline()) for f in h_files]

    vprint("---Calibrating---\nDimensions (cm): {1}\nImages and Heights: {0}"\
        .format(zip(files, heights), dim))

    assert len(files) == len(heights)
    cal_points = np.zeros((len(files), 2))
    cal_pointsY = np.zeros((len(files), 2))
    cal_pointsX = np.zeros((len(files), 2))
    for i, f in enumerate(files):
        vprint("\nImg: {0}".format(f))

        (y, x, g) = grab_pix_dim(f, direc=cal_dir)
        cm_per_px = (dim[0]/y + dim[1]/x)/2.
        cm_per_pxY, cm_per_pxX = dim[0]/y, dim[1]/x

        vprint("cm/px (Y) = {0}, cm/px (X) = {1}"\
            .format(cm_per_pxY, cm_per_pxX))
        vprint("Results: {0}x{1} (px), height: {2} (cm) -> cm/px (avg) = {3}"\
            .format(y, x, heights[i], cm_per_px))

        cal_points[i] = [heights[i], cm_per_px]
        cal_pointsY[i] = [heights[i], cm_per_pxY]
        cal_pointsX[i] = [heights[i], cm_per_pxX]

    cal_line = np.polyfit(cal_points[:,0], cal_points[:,1], 1)
    cal_lineY = np.polyfit(cal_pointsY[:,0], cal_pointsY[:,1], 1)
    cal_lineX = np.polyfit(cal_pointsX[:,0], cal_pointsX[:,1], 1)

    if args.disp:
        x = np.linspace(0, 40, 1000)
        y1 = cal_line[0]*x + cal_line[1]
        y2 = cal_lineY[0]*x + cal_lineY[1]
        y3 = cal_lineX[0]*x + cal_lineX[1]
        plt.plot(x, y1)
        plt.plot(x, y2)
        plt.plot(x, y3)
        plt.legend(["cm/px (avg)", "cm/px (Y)", "cm/px (X)"])
        plt.plot(cal_points[:,0], cal_points[:,1], 'o')
        plt.xlabel('Box Height')
        plt.ylabel('cm / px')
        plt.show()


    vprint("Cal Line = %fx + %f"%(cal_line[0], cal_line[1]))
    vprint("---END Calibrating---\n")
    return cal_line

def grab_coords(img, deriv, axis=0, thresh=500):
    img_copy = np.copy(img)
    img_copy[abs(deriv) < thresh] = 0
    img_copy = np.clip(img_copy, 0, 1)
    # disp_image([sk.color.rgb2grey(img_copy)], cmap='gray', flag=True)
    vals = np.sum(np.sum(img_copy, axis=(axis+1)%2), axis=1)
    vals[:len(img)*.1] = 0
    vals[-len(img)*.1:] = 0
    Nhalf = len(vals) / 2.
    return [np.argmax(vals[:Nhalf]), Nhalf+np.argmax(vals[Nhalf:])], vals

def grab_pix_dim(file_name, direc=CUR_DIR):
    im, height = load_image(file_name, direc=direc)
    im = im.astype('int32')
    dx = ndimage.sobel(im, axis=0, mode='constant')
    dy = ndimage.sobel(im, axis=1, mode='constant')

    dx_gray = sk.color.rgb2grey(dx)
    dy_gray = sk.color.rgb2grey(dy)
    disp_image([dx_gray, dy_gray, dx_gray+dy_gray], cmap='gray')
    # y_coord, y_vals = grab_coords(abs(dx)+abs(dy), dx, axis=0)
    # x_coord, x_vals = grab_coords(abs(dy)+abs(dx), dy, axis=1, thresh=400)
    # vprint("Y-Coord2 (px): {0}, X-Coord2 (px): {1}".format(y_coord, x_coord))

    y_coord, y_vals = grab_coords(im, dx, axis=0)
    x_coord, x_vals = grab_coords(im, dy, axis=1, thresh=400)
    y_length_pix = abs(y_coord[1] - y_coord[0])
    x_length_pix = abs(x_coord[1] - x_coord[0])
    grad = get_grad(im, y_vals, x_vals)

    vprint("Y-Coord (px): {0}, X-Coord (px): {1}".format(y_coord, x_coord))
    return y_length_pix, x_length_pix, grad

def grab_dim(file_name, cal_line, direc=CUR_DIR):
    print("===Find Dimensions of {0}===".format(file_name))
    im, height = load_image(file_name, direc=direc)
    y_length_pix, x_length_pix, grad = grab_pix_dim(file_name, direc=direc)
    cm_per_px = cal_line[0]*(height) + cal_line[1]
    y_length_cm = y_length_pix * cm_per_px
    x_length_cm = x_length_pix * cm_per_px

    vprint("Height: {0} -> cm/px: {1}".format(height, cm_per_px))
    vprint("Dimensions: {0}x{1} (cm)".format(y_length_cm, x_length_cm))
    vprint("Height: {0} (cm)".format(height))
    disp_image([im, grad])

    return y_length_cm, x_length_cm, height

### MAIN ###
def wait_for_image(start_files, direc=CUR_DIR):
    vprint("Start Files: {0}".format(start_files))

    new_files = [f for f in listdir(direc) if isfile(join(direc, f)) and 'jpg' in f]

    while len(new_files) - len(start_files) < 1:
        new_files = [f for f in listdir(direc) if isfile(join(direc, f)) and 'jpg' in f]
        vprint("\nDetecting...")
        vprint("Current Files: {0}".format(new_files))
        time.sleep(3)

    time.sleep(5)
    diff = set(new_files) - set(start_files)
    vprint("\nDifferent Files: {0}".format(diff))
    return diff, direc

# write dimensions and size of box from image file to a text file
def write_dim(y, x, h, file_name):
    f = open("out_dimensions_record.txt", 'a')
    s = "\n%s: %.2fx%.2f (cm), %.2f (cm): "%(file_name, y, x, h)
    size = y * x * h
    if size < SMALL_THRESH:
        s += "SMALL (0)"
    elif size < MEDIUM_THRESH:
        s += "MEDIUM (1)"
    else:
        s += "LARGE (2)"
    f.write(s)
    f.close()

if __name__ == '__main__':
    print "==Auto Box Dimensions Measurement=="
    vprint("Verbose ON")
    vprint("Display ON\n", flag=args.disp)

    if args.action == "calibrate":
        cal_f = open(CAL_DIR+cal_text_name, 'w')
        cal_f.write("DIMENSIONS:{0},{1}".format(args.ylen, args.xlen))
        cal_f.close()
        counter = 0
        start_files = [f for f in listdir(CUR_DIR) if isfile(join(CUR_DIR, f)) and 'jpg' in f]
        while True:
            files, direc = wait_for_image(start_files)
            counter += len(files)
            time.sleep(3)
            for f in files:
                os.rename(direc+f, CAL_DIR+f)
                os.rename(direc+f[:-4] + ".txt", CAL_DIR+f[:-4] + ".txt")

    elif args.action == 'poll':
        cal_line = calibrate(CAL_DIR)
        start_files = [f for f in listdir(CUR_DIR) if isfile(join(CUR_DIR, f)) and 'jpg' in f]

        while True:
            files, direc = wait_for_image(start_files)
            for f in files:
                print "\nFind Dimensions of {0}".format(f)
                (y, x, h) = grab_dim(f, cal_line)
                write_dim(y, x, h, f)
                start_files.append(f)

    elif args.action == "find_dim":
        cal_line = calibrate(CAL_DIR)
        (y, x, h) = grab_dim(args.input, cal_line)
        write_dim(y, x, h, args.input)