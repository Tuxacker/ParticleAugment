import argparse
import base64
import glob
import os
import re
import sys
import shutil
import json
from pathlib import Path



from PIL import Image
from tqdm import tqdm

import numpy as np

from pycocotools import mask as cocomask
from nuimages import NuImages
import cv2

np.set_printoptions(precision=2)
np.set_printoptions(threshold=sys.maxsize)

#import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
#from matplotlib.collections import PatchCollection

def interp_line(a, b, factor):
    return a + (b - a) * factor

def interp_bezier3(vertices, factor):
    assert vertices.shape[0] == 3
    p1 = interp_line(vertices[0], vertices[1], factor)
    p2 = interp_line(vertices[1], vertices[2], factor)
    return interp_line(p1, p2, factor)

def interp_bezier4(vertices, factor):
    assert vertices.shape[0] == 4
    a = interp_line(vertices[0], vertices[1], factor)
    b = interp_line(vertices[1], vertices[2], factor)
    c = interp_line(vertices[2], vertices[3], factor)
    p1 = interp_line(a, b, factor)
    p2 = interp_line(b, c, factor)
    return interp_line(p1, p2, factor)

#def interp_bezier(vertices, factor):
#    if vertices.shape[0] == 4:
#        return interp_bezier4(vertices, factor)
#    elif vertices.shape[0] == 3:
#        return interp_bezier3(vertices, factor)
#    else:
#        return ValueError("Invalid Bezier curve length: {}".format(vertices.shape[0]))

def to_full_vertices(vertices, types, points_per_curve=10):
    out_vertices = []
    in_curve = False
    v_cache = []
    for v, t in zip(vertices, types):
        if t == "L":
            if in_curve:
                v_cache.append(v)
                interp_locations = np.linspace(0.0, 1.0, points_per_curve)
                for i in interp_locations[1:]:
                    out_vertices.append(interp_bezier4(np.array(v_cache), i))
                in_curve = False
                v_cache = [v]
            else:
                out_vertices.append(v)
                v_cache = [v]
        elif t == "C":
            v_cache.append(v)
            in_curve = True
    if len(v_cache) > 2:
        v_cache.append(vertices[0])
        interp_locations = np.linspace(0.0, 1.0, points_per_curve)
        for i in interp_locations[1:-1]:
            out_vertices.append(interp_bezier4(np.array(v_cache), i))
    return out_vertices


def bdd2coco_detection(id_dict, scenes, fn, image_dir, out_img):

    images = list()
    annotations = list()

    image_paths = glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True)
    image_names = [ip.split("/")[-1] for ip in image_paths]

    counter = 0
    ann_counter = 777777
    for s in tqdm(scenes):
        for i in s:
            counter += 1
            image = dict()

            print(i)

            full_path = image_paths[image_names.index(i['name'])]
            
            image['file_name'] = i['name']
            image['height'] = 720
            image['width'] = 1280

            image['id'] = counter

            empty_image = True

            if not 'labels' in i.keys() or i['labels'] is None:
                continue

            for label in i['labels']:
                #fig, ax = plt.subplots()
                #img = Image.open(full_path)
                #vertices = np.array(label['poly2d'][0]['vertices'])
                #types = label['poly2d'][0]['types']
                #vertices = to_full_vertices(vertices, types)
                #ax.imshow(img)
                #polygon = [Polygon(vertices, True)]
                #p = PatchCollection(polygon, alpha=0.4)
                #ax.add_collection(p)
                #fig.savefig("test_seg_mask.png")
                annotation = dict()
                ann_counter += 1
                category = label['category']
                poly2d = label['poly2d']
                if category in id_dict.keys() and poly2d is not None:
                    empty_image = False
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = image['id']
                    annotation['category_id'] = id_dict[category]
                    annotation['ignore'] = 0
                    annotation['id'] = ann_counter# label['id']
                    polygons = []
                    xmin = []
                    xmax = []
                    ymin = []
                    ymax = []
                    for poly in poly2d:
                        vertices = np.array(to_full_vertices(np.array(poly['vertices']), poly['types']))
                        xmin.append(np.min(vertices[:, 0]))
                        xmax.append(np.max(vertices[:, 0]))
                        ymin.append(np.min(vertices[:, 1]))
                        ymax.append(np.max(vertices[:, 1]))
                        polygon_items = ["{:.2f}".format(pv) for pv in vertices.flatten()]
                        polygon_string = "[" + ", ".join(polygon_items) + "]"
                        polygons.append(polygon_string)
                    xmin = np.min(xmin)
                    xmax = np.max(xmax)
                    ymin = np.min(ymin)
                    ymax = np.max(ymax)
                    annotation['segmentation'] = polygons
                    annotation['bbox'] = "[{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(xmin, ymin, xmax-xmin, ymax-ymin)
                    annotation['area'] = "{:.2f}".format(float((xmax - xmin) * (ymax - ymin)))
                    #print(annotation)
                    annotations.append(annotation)

            if empty_image:
                continue
            else:
                shutil.copy(full_path, out_img)

            #print(image)

            images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict).replace("\"[", "[").replace("]\"", "]").replace("},", "},\n")
    json_string = re.subn(r'\"area\":\s+\"([0-9]*\.[0-9]*)\"', r'"area": \1', json_string)[0]
    with open(fn, "w") as file:
        file.write(json_string)

def rle_to_polygon(rle):
    mask_arr = cocomask.decode(rle)

    contours, _ = cv2.findContours(mask_arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        # Valid polygons have >= 8 coordinates (4 points)
        if contour.size >= 8:
            segmentation.append(contour.flatten().tolist())
    RLEs = cocomask.frPyObjects(segmentation, mask_arr.shape[0], mask_arr.shape[1])
    RLE = cocomask.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = cocomask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(mask_arr)

    return segmentation #, [x, y, w, h], area

def nuimages2coco(dataroot, version, tag="val"):

    label_dir = os.path.join(dataroot, "COCO")
    image_dir = os.path.join(dataroot, "COCO", tag)
    Path(image_dir).mkdir(parents=True, exist_ok=True)


    label_dict = dict()

    label_dict["categories"] = [
            {"supercategory": "none", "id": 1, "name": "person"},
            {"supercategory": "none", "id": 2, "name": "bicycle"},
            {"supercategory": "none", "id": 3, "name": "car"},
            {"supercategory": "none", "id": 4, "name": "bus"},
            {"supercategory": "none", "id": 5, "name": "truck"},
            {"supercategory": "none", "id": 6, "name": "motorcyle"},
            {"supercategory": "none", "id": 7, "name": "trailer"},
            {"supercategory": "none", "id": 8, "name": "animal"}
        ]

    nuim = NuImages(dataroot=dataroot, version=version)

    nuim_categories = nuim.category

    category_map = dict()

    for cat in nuim_categories:
        if "vehicle" in cat["name"]:
            if cat["name"] == "vehicle.ego":
                pass
            elif "vehicle.bus" in cat["name"]:
                category_map[cat["token"]] = 4
            elif cat["name"] == "vehicle.trailer":
                category_map[cat["token"]] = 7
            elif cat["name"] == "vehicle.truck":
                category_map[cat["token"]] = 5
            elif cat["name"] == "vehicle.bicycle":
                category_map[cat["token"]] = 2
            elif cat["name"] == "vehicle.motorcycle":
                category_map[cat["token"]] = 6
            else:
                category_map[cat["token"]] = 3
        elif "human" in cat["name"]:
            category_map[cat["token"]] = 1
        elif cat["name"] == "animal":
            category_map[cat["token"]] = 8

    images = list()
    annotations = list()

    img_id = 1000

    ann_id = 1000000

    for i, sample in enumerate(tqdm(nuim.sample)):
        #if i > 50:
        #    break
        token_ann = sample["token"]
        token_img = sample["key_camera_token"]

        objects, _ = nuim.list_anns(token_ann, verbose=False)

        has_valid_annotations = False

        for obj_token in objects:
            obj = nuim.get("object_ann", obj_token)
            if obj["category_token"] in category_map.keys() and obj["mask"] is not None:
                has_valid_annotations = True
                break
        
        if not has_valid_annotations:
            continue

        image_metadata = nuim.get("sample_data", token_img)
        #new_filename = os.path.join(image_dir, "{:06d}.jpg".format(img_id))
        shutil.copyfile(os.path.join(dataroot, image_metadata["filename"]), new_filename)

        image = dict()
        image["file_name"] = "{:06d}.jpg".format(img_id)
        image["height"] = image_metadata["height"]
        image["width"] = image_metadata["width"]

        image["id"] = img_id
        img_id += 1

        images.append(image)

        for obj_token in objects:
            obj = nuim.get("object_ann", obj_token)
            if obj["category_token"] in category_map.keys() and obj["mask"] is not None:
                annotation = dict()
                annotation["iscrowd"] = 0
                annotation["image_id"] = image["id"]
                annotation["category_id"] = category_map[obj["category_token"]]
                annotation["ignore"] = 0
                annotation["id"] = ann_id
                ann_id += 1
                try:
                    annotation["segmentation"] = rle_to_polygon({"size": obj["mask"]["size"], "counts": base64.b64decode(obj["mask"]["counts"]).decode("ascii")})
                    annotation["bbox"] = [obj["bbox"][0], obj["bbox"][1], obj["bbox"][2] - obj["bbox"][0], obj["bbox"][3] - obj["bbox"][1]]
                    annotation["area"] = 0
                    annotations.append(annotation)
                except IndexError:
                    print("Found invalid mask, skipping...")

    label_dict["images"] = images
    label_dict["annotations"] = annotations
    label_dict["type"] = "instances"

    #json_string = json.dumps(attr_dict).replace("\"[", "[").replace("]\"", "]").replace("},", "},\n")
    #json_string = re.subn(r'\"area\":\s+\"([0-9]*\.[0-9]*)\"', r'"area": \1', json_string)[0]
    with open(os.path.join(label_dir, tag + ".json"), "w") as file:
        file.write(json.dumps(label_dict).replace("},", "},\n"))



if __name__ == '__main__':

    source = "nuimages"

    if source == "nuimages":
        dataroot = "/share/Projects/Datasets/nuimages"
        version = "v1.0-val"

        nuimages2coco(dataroot, version)
    
    else:

        prefix = "train"

        image_dir="/share/Projects/Datasets/BDD100K/bdd100k_seg_track_20_images/bdd100k/images/seg_track/" + prefix
        label_dir="/share/Projects/Datasets/BDD100K/bdd100k_seg_track_20_labels_release_mots2020/bdd100k/labels/seg_track/polygons/" + prefix
        save_path="/share/Projects/Datasets/BDD100K/BDD100K_COCO/"

        attr_dict = dict()
        attr_dict["categories"] = [
            {"supercategory": "none", "id": 1, "name": "person"},
            {"supercategory": "none", "id": 2, "name": "rider"},
            {"supercategory": "none", "id": 3, "name": "car"},
            {"supercategory": "none", "id": 4, "name": "bus"},
            {"supercategory": "none", "id": 5, "name": "truck"},
            {"supercategory": "none", "id": 6, "name": "bike"},
            {"supercategory": "none", "id": 7, "name": "motor"}
        ]

        attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

        train_labels = []

        # create BDD training set detections in COCO format
        print('Loading training scenes...')
        files = glob.glob(os.path.join(label_dir, "*.json"))
        for file in tqdm(files):
            with open(file) as f:
                train_labels.append(json.load(f)[:10])
        print('Converting {} training scenes...'.format(len(train_labels)))

        #print(json.dumps(train_labels[0], indent=4))

        out_fn = os.path.join(save_path, "bdd100k_coco_{}.json".format(prefix))
        out_img = os.path.join(save_path, prefix)
        if not os.path.exists(out_img):
            os.mkdir(out_img)
        bdd2coco_detection(attr_id_dict, train_labels, out_fn, image_dir, out_img)

        #print('Loading validation set...')
        # create BDD validation set detections in COCO format
        #with open(os.path.join(label_dir,
        #                       'det_v2_val_release.json')) as f:
        #    val_labels = json.load(f)
        #print('Converting validation set...')

        #out_fn = os.path.join(save_path,
        #                      'bdd100k_labels_images_det_coco_val.json')
        #bdd2coco_detection(attr_id_dict, val_labels, out_fn)