#! /usr/bin/env python3

import json
import os
import shutil
import numpy as np
from osgeo import osr, ogr, gdal
from tqdm import tqdm
import time
import cv2
import sys
import torch
import pygeohash as pgh
import datetime
from dateutil import parser


class Detector:
    def __init__(self, weights, device='cpu', conf_thr=0.25, iou_thr=0.45):

        # Loading Model
        self.model = torch.hub.load("yolov5s", 'custom', path=weights, source='local')  # local repo

        # Configuring Model
        if device == 'cpu':
            self.model.cpu()  # .cpu() ,or .cuda()
        else:
            self.model.cuda()
        self.model.conf = conf_thr  # NMS confidence threshold
        self.model.iou = iou_thr  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        self.model.max_det = 100  # maximum number of detections per image
        self.model.amp = False  # Automatic Mixed Precision (AMP) inference

    def detect(self, file_inp):

        # Reading Image
        im = cv2.imread(file_inp)
        im0 = im.copy()
        results = self.model(im, size=640)
        results_df = results.pandas().xyxy[0]

        # Process detections
        det_bb_list = []
        if len(results_df):
            for index, row in results_df.iterrows():
                # Centroid Coordinates of detected object
                cx = int((row['xmin'] + row['xmax']) / 2.0)
                cy = int((row['ymin'] + row['ymax']) / 2.0)
                det_bb_list.append((cx, cy))
        return det_bb_list


def build_transform_inverse(dataset, EPSG):
    source = osr.SpatialReference(wkt=dataset.GetProjection())
    target = osr.SpatialReference()
    target.ImportFromEPSG(EPSG)
    return osr.CoordinateTransformation(source, target)


def conv_to_jpg(fl_path, conv_img_dir):
    """Convert Geotiff to JPEG Images"""

    tiff_dataset_temp = gdal.Open(fl_path, gdal.GA_ReadOnly)
    trns_inv = build_transform_inverse(tiff_dataset_temp, 4326)
    fl_name = os.path.basename(fl_path).split('.')[0]
    nBand, height, width = tiff_dataset_temp.RasterCount, tiff_dataset_temp.RasterYSize, tiff_dataset_temp.RasterXSize
    img_data = tiff_dataset_temp.ReadAsArray(0, 0, width, height)
    second_min = np.amin(img_data[img_data != np.amin(img_data)])

    img_data[img_data != -32768.0] -= second_min
    to_add = round(img_data.max() / 5)
    img_data[img_data != -32768.0] += to_add
    img_data[img_data == -32768.0] = 0
    imax_t = img_data.max()
    img_data *= (255 / imax_t)
    conv_img_path = os.path.join(conv_img_dir, f'{fl_name}.jpg')
    cv2.imwrite(conv_img_path, img_data)
    return conv_img_path, tiff_dataset_temp, trns_inv


def seg_sea_area(im_gray):
    """Segment Sea Area from the image"""
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)
    (thresh, mask1) = cv2.threshold(im_gray, 10, 255, cv2.THRESH_BINARY)
    mask1 = cv2.bitwise_not(mask1)

    kernel = np.ones((5, 5), np.uint8)
    im_bw = cv2.dilate(im_bw, kernel, iterations=1)
    mask = np.zeros(im_bw.shape, np.uint8)
    # im_inv = cv2.bitwise_not(im_bw)

    contours, hier = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print("No of contours :", len(contours))
    n = 0

    for cnt in contours:
        if cv2.contourArea(cnt) > 10000:
            # print(n)
            n += 1
            # cv2.drawContours(im_gray, [cnt], 0, (0, 255, 0), 5)
            cv2.drawContours(mask, [cnt], 0, 255, -1)

    mask_tot = cv2.bitwise_not(mask + mask1)
    kernel = np.ones((10, 10), np.uint8)
    sea_mask = cv2.erode(mask_tot, kernel, iterations=1)
    return sea_mask


def crop_im(img_inp, crp_dir, crp_sz=2000, thr=0.35):
    """Apply sea mask and cropping into small crops"""

    sea_mask = seg_sea_area(img_inp)
    cv2.imwrite('img2.jpg', img_inp)
    v_size, h_size = img.shape
    v_crp_nos = round(v_size / crp_sz)
    h_crp_nos = round(h_size / crp_sz)

    # Finding points to divide image into equal sized segments
    v_segs = [v_size // v_crp_nos + (1 if x < v_size % v_crp_nos else 0) for x in range(v_crp_nos)]
    h_segs = [h_size // h_crp_nos + (1 if x < h_size % h_crp_nos else 0) for x in range(h_crp_nos)]

    v_segs1 = [sum(v_segs[:i]) for i in range(len(v_segs) + 1)]
    h_segs1 = [sum(h_segs[:i]) for i in range(len(h_segs) + 1)]
    num = 0
    for i in tqdm(range(len(h_segs1) - 1)):
        for j in range(len(v_segs1) - 1):
            num += 1
            ymin = v_segs1[j]
            ymax = v_segs1[j + 1]
            xmin = h_segs1[i]
            xmax = h_segs1[i + 1]
            img_crop = img_inp[ymin:ymax, xmin:xmax]

            mask_crop = sea_mask[ymin:ymax, xmin:xmax]
            mask_flat = mask_crop.flatten()
            if sum(mask_flat) > (thr * len(mask_flat)):
                img_masked = cv2.bitwise_and(mask_crop, img_crop)
                cv2.imwrite(os.path.join(crp_dir,
                                         f'x1_{xmin}_x2_{xmax}_y1_{ymin}_y2_{ymax}.jpg'), img_crop)
    print(f"Num of crops total : ", num)
    return sea_mask


def detect_vessels(vessel_detctr, imq_path, sea_mask):
    """Function to detect vessels using YOLO Model"""
    im_name = os.path.basename(img_pth)
    # Finding full image co-ordinate of left top corner of the crop
    crp_minx, crp_miny = int(im_name.split('_')[1]), int(im_name.split('_')[5])
    # YOLO Detection
    detctd_vessls_crp = vessel_detctr.detect(imq_path)
    # Mapping to full image co-ordinates
    detctd_vessls = get_whole_img_co_ordnts(crp_minx, crp_miny, detctd_vessls_crp, sea_mask)
    return detctd_vessls


def get_whole_img_co_ordnts(crp_minx, crp_miny, detctd_vessls_crp, sea_mask):
    """Function to convert Crop image co-ordinates to Whole image co-ordinates"""
    detctd_vessls_full = []
    for det_bb in detctd_vessls_crp:
        bb_full = (int(det_bb[0] + crp_minx), int(det_bb[1] + crp_miny))
        msk_crp = sea_mask[(bb_full[1] - 5):(bb_full[1] + 5), (bb_full[0] - 5):(bb_full[0] + 5)]
        if sum(msk_crp.flatten()) > 70:
            detctd_vessls_full.append(bb_full)
    return detctd_vessls_full


def pixel_to_world(geo_matrix, x, y):
    ul_x = geo_matrix[0]
    ul_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    y_dist = geo_matrix[5]
    _x = x * x_dist + ul_x
    _y = y * y_dist + ul_y
    return _x, _y


def find_spatial_coordinate_from_pixel(dataset, transform, x, y):
    world_x, world_y = pixel_to_world(dataset.GetGeoTransform(), x, y)
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(world_x, world_y)
    point.Transform(transform)
    return point.GetX(), point.GetY()


if __name__ == "__main__":
    strt_time = time.time()
    model_path = './runs/train/Model12/weights/best.pt'
    vessel_detctr = Detector(model_path, conf_thr=0.25, iou_thr=0.45)
    f_types = [('tif Files', '*.tif')]
    tiff_path = sys.argv[1]
    seg_size = 1000

    conv_img_dir = './tmp/conv_out'
    crp_dir = './tmp/crop_imgs'
    prd_out_dir = './tmp/pred_out'

    # Clear directories
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    os.makedirs(conv_img_dir)
    os.makedirs(crp_dir)
    os.makedirs(prd_out_dir)

    print("Conversion Started...")
    conv_img_path, tiff_dataset_temp, trns_inv = conv_to_jpg(tiff_path, conv_img_dir)
    print("Conversion completed.", "Time Taken : ", time.time() - strt_time, '\n')

    img = cv2.imread(conv_img_path, 0)
    print("Cropping Started...")
    sea_mask = crop_im(img, crp_dir, seg_size)
    print("Cropping completed.", "Time Taken : ", time.time() - strt_time, '\n')

    print("YOLO Detection Started...")
    final_detctd_vessls = []
    for im in tqdm(os.listdir(crp_dir)):
        img_pth = os.path.join(crp_dir, im)
        # "yolo detection....."
        crp_detctd_vessls = detect_vessels(vessel_detctr, img_pth, sea_mask)
        # print("detected vessels :", crp_detctd_vessls)
        # print(im, ":", len(crp_detctd_vessls))
        final_detctd_vessls.extend(crp_detctd_vessls)
    print("YOLO Detection completed.", "Time Taken : ", time.time() - strt_time, '\n')

    print("Annotation Started...")
    img_clr = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    final_detctd_vessls_info = []
    count = 0
    for vess_bb in tqdm(final_detctd_vessls):
        vessls_info_dict = {}
        spatial_co_ords = find_spatial_coordinate_from_pixel(tiff_dataset_temp, trns_inv, vess_bb[0], vess_bb[1])
        uuid = pgh.encode(spatial_co_ords[0], spatial_co_ords[1], precision=5)
        time_stamp = time.time()
        tz = datetime.timezone.utc
        ft = "%Y-%m-%dT%H:%M:%S%z"
        t = datetime.datetime.now(tz=tz).strftime(ft)
        epoch = parser.parse(t).timestamp()
        count += int("00")+1
        vessls_info_dict['track_name'] = f"SAR{uuid}{epoch % 100000}{count}"
        vessls_info_dict['track_id'] = f"SAR{uuid}"
        vessls_info_dict['lat'] = spatial_co_ords[0]
        vessls_info_dict['long'] = spatial_co_ords[1]
        vessls_info_dict['time_stamp'] = time_stamp

        final_detctd_vessls_info.append(vessls_info_dict)

      

    # file name is mydata
    with open("vessel_info_out.json", "w") as final:
        json.dump(final_detctd_vessls_info, final)


    print("Annotation Completed. Time Taken : ", time.time() - strt_time, '\n')
    print("Detected Vessels Lat Lon :", final_detctd_vessls_info)
