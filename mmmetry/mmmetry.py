import os
import os.path
import math
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image
from PIL import ImageDraw
import warnings
warnings.simplefilter('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import albumentations as A
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import segmentation_models_pytorch as smp
import gdown
import zipfile

def main(root, path, is_wsi, scale, fib_det_th):
    
    fascicle_seg_weights_path = f'{root}/weights/mask_rcnn_20220724.pt'
    fiber_det_weights_path = f'{root}/weights/mobilenet_20220704.pt'
    myelin_seg_weights_path = f'{root}/weights/Deeplab_20220608.pt'
    
    weight_paths = [fascicle_seg_weights_path, fiber_det_weights_path, myelin_seg_weights_path, f'{root}/weights/gmm_classifier.pkl']

    for path_wt in weight_paths:
        if not os.path.exists(path_wt):
            url = 'https://drive.google.com/uc?id=1Fjhv9220C4axKodhwgDPZGHCIWBLJj30'
            output = f'{root}/weights/hifuku_wt.zip'
            os.makedirs(f'{root}/weights', exist_ok=True)
            gdown.download(url, output, quiet=False)

            # Extract the zip file
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(f'{root}')

            # Delete the zip file after extraction
            os.remove(output)
            break
        

    # set parameters
    margin = 25
    crop_size = 200
    color_o = (255, 73, 0)
    color_i = (255, 182, 0)
    # make a directory to save results
    image_id = os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]
    save_dir= f'{root}/results/{image_id}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/fascicles', exist_ok=True)
    # prepare for fascicle segmentation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pl.seed_everything(0)
    n_fascicle_seg_classes = 2
    model = get_instance_segmentation_model(n_fascicle_seg_classes).eval()
    model.load_state_dict(torch.load(fascicle_seg_weights_path, device))

    size = (2500, 2500)
    img_l_PIL = Image.open(path)
    img_l_np = np.array(img_l_PIL)
    original_size = (img_l_np.shape[0], img_l_np.shape[1])
    img_s_PIL = img_l_PIL.resize(size)
    img_s_np = np.array(img_s_PIL)

    data_nerve = pd.DataFrame({
        'id': image_id,
        'is_wsi': is_wsi,
        'x_px': original_size[0],
        'y_px': original_size[1],
        'scale': scale,
    }, index=[0])

    # segment fascicles
    transforms = torchvision.transforms.ToTensor()
    model.to(device)
    x1 = transforms(img_s_np)
    y1 = model(x1.unsqueeze(0).to(device))[0]
    
    # visualize the segmented fascicles
    merged = merge_prediction(x1, y1)
    merged_PIL = Image.fromarray(merged)
    merged_PIL_resized = merged_PIL.resize((int(original_size[1]), int(original_size[0])))
    merged_PIL_resized.save(f'{save_dir}/masked_fascicles.jpg')
    
    # crop and mask fascicles
    if is_wsi == True:
        masked_images = mask_fascicle(path, y1)
    else:
        masked_images = [img_l_np]
    # split fascicles
    splitted_fascicles = split_fascicle(masked_images)
    # detect nerve fibers
    fibers_s = []
    xy_fascicles = []
    for i in range(len(splitted_fascicles)):
        fibers = []
        fiber_images, boxes_s = detect_fiber(splitted_fascicles[i]['image'], fiber_det_weights_path, margin, threshold=fib_det_th)
        boxes_large = get_boxes_large(splitted_fascicles[i], boxes_s, crop_size, margin)
        for box in boxes_large:
            fibers.append(masked_images[i][math.floor(box[1]):math.ceil(box[3]), math.floor(box[0]):math.ceil(box[2])])
        fibers_s.append(fibers)
        xy_fascicles.append(boxes_large)
        image_PIL = visualize_detection(masked_images[i], boxes_large, color_o, margin)
        Image.fromarray(masked_images[i]).save(f'{save_dir}/fascicles/fascicle_{i:03}.jpg')
        image_PIL.save(f'{save_dir}/fascicles/fascicle_{i:03}_bbox.jpg')
    # set segmentation model for detected fibers
    deeplab = Deeplab(n_classes=3).eval()
    deeplab.to(device)
    deeplab.load_state_dict(torch.load(myelin_seg_weights_path))
    # segment fibers
    cols  = [
        'area_out', 
        'area_in', 
    ]
    df = pd.DataFrame([], columns=cols)
    images = []
    images_cnt = []
    for n, fibers in enumerate(fibers_s):
        for fiber, xy in zip(fibers, xy_fascicles[n]):
            segmented_fiber = get_segmentation(fiber, deeplab, scale, color_o, color_i, device=device)
            images.append(segmented_fiber['img'])
            images_cnt.append(segmented_fiber['img_cnt'])
            segmented_fiber['df']['n_fas'] = n
            segmented_fiber['df']['xmin'] = xy[0]
            segmented_fiber['df']['ymin'] = xy[1]
            segmented_fiber['df']['xmax'] = xy[2]
            segmented_fiber['df']['ymax'] = xy[3]
            df = pd.concat([df, segmented_fiber['df']], ignore_index=True)
            
    # visualize fibers
    fig = tile_images(fibers_s[0], 10, 10)
    plt.savefig(f'{save_dir}/fibers.jpg')
    fig = tile_images(images_cnt, 10, 10)
    plt.savefig(f'{save_dir}/fibers_annotated.jpg')
    
    # calculate parameters in each fascicule
    threshold = 0.8
    fas_area = []
    masks_np = y1['masks'].mul(255).permute(0, 2, 3, 1).squeeze(3).byte().cpu().numpy()
    for c in range(len(masks_np)):
        if y1['scores'][c] > threshold:
            c_bool = (masks_np[c] > 127)
            mask_np = (c_bool*255).astype(np.uint8)
            mask_PIL = Image.fromarray(mask_np).resize(original_size)
            mask_bin = np.array(mask_PIL) > 127
            area_mm2 = np.count_nonzero(mask_bin) * scale * scale / 1000000
            fas_area.append(area_mm2)

    if is_wsi == True:
        total_area = sum(fas_area)
    else:
        total_area = (original_size[0] - margin) * (original_size[1] - margin) * scale * scale / 1000000
        fas_area = [total_area]

    total_density = len(df) / total_area
    
    densities = []
    n_fib_s = []
    for n, area in enumerate(fas_area):
        n_fib = len(df[df['n_fas'] == n])
        n_fib_s.append(n_fib)
        density = n_fib / area
        densities.append(density)

    data_fas = pd.DataFrame(
        data={
            'n_fas': range(len(fas_area)),
            'n_fib': n_fib_s,
            'area': fas_area,
            'density': densities
            })

    data_nerve['total_fas'] = len(data_fas)
    data_nerve['n_fibers'] = len(df)
    data_nerve['total_area'] = total_area
    data_nerve['total_density'] = total_density
    data_nerve['std_density'] = data_fas.describe()['density']['std']
    data_nerve.to_csv(f'{save_dir}/data_nerve.csv')
    data_fas.to_csv(f'{save_dir}/data_fas.csv')
    df.to_csv(f'{save_dir}/data_fib.csv')


class Net(pl.LightningModule):
    def __init__(self, n_feature=1024, n_class=2):
        super().__init__()
        self.faster_rcnn = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(n_feature, n_class)

    def forward(self, x, t=None):
        if self.training:
            return self.faster_rcnn(x, t)
        else:
            return self.faster_rcnn(x)


class Deeplab(pl.LightningModule):
    def __init__(self, in_channels=3, n_classes=3):
        super().__init__()
        self.net = smp.DeepLabV3Plus('resnet18', in_channels=in_channels, classes=n_classes, encoder_weights='imagenet')

    def forward(self, x):
        return self.net(x)


def get_instance_segmentation_model(n_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, n_classes)
    return model


def merge_prediction(x, y, threshold=0.95):
    masks_np = y['masks'].mul(255).permute(0, 2, 3, 1).byte().cpu().numpy()
    colors = sns.color_palette(palette='hls', n_colors=len(masks_np))
    color_mask = np.zeros((masks_np.shape[1], masks_np.shape[2], 3))
    for c in range(len(masks_np)):
        if y['scores'][c] > threshold:
            c_bool = (masks_np[c] >= 127)
            color_mask[:, :, 0] += (c_bool[:, :, 0] * colors[c][0])
            color_mask[:, :, 1] += (c_bool[:, :, 0] * colors[c][1])
            color_mask[:, :, 2] += (c_bool[:, :, 0] * colors[c][2])
    color_mask  = (color_mask * 255).astype(np.uint8)
    img = x.mul(255).permute(1, 2, 0).byte().numpy().astype(np.uint8)
    merged = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    return merged


def mask_fascicle(path, y1, threshold=0.8):
    masked_images = []
    for i, score in enumerate(y1['scores']):
        if score > threshold:
            img_l_PIL = Image.open(path)
            img = np.array(img_l_PIL)
            original_size = img.shape
            mask_pt = y1['masks'][i]
            mask_np = mask_pt.mul(255).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask_np, (original_size[1], original_size[0]))
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            box = [xmin, ymin, xmax, ymax]
            _, bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            img[bin==0] = [0, 0, 0]
            img_masked = img[box[1]:box[3], box[0]:box[2], :]
            masked_images.append(img_masked)
    return masked_images


def split_fascicle(masked_images, crop_size=200, margin=25):
    splitted_fascicles = []
    for img_np in masked_images:
        splitted_fascicle = []
        width = img_np.shape[1]
        height = img_np.shape[0]
        n_w = int(width / crop_size) +1
        n_h = int(height / crop_size) +1
        # padding
        pad_w = crop_size * n_w - width
        pad_h = crop_size * n_h - height
        img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, (0, 0, 0))

        for h in range(n_h):
            for w in range(n_w):
                split = A.Compose(
                    [
                    A.Resize(height=n_h*crop_size+2*margin, width=n_w*crop_size+2*margin),
                    A.Crop(w*crop_size, h*crop_size, (w+1)*crop_size+2*margin, (h+1)*crop_size+2*margin)
                    ],
                )
                splitted = split(image=img_np)
                splitted_fascicle.append(splitted['image'])
        fas_dict = {'image': splitted_fascicle, 'n_w': n_w, 'n_h':n_h, 'original_size': (width, height)}
        splitted_fascicles.append(fas_dict)
    return splitted_fascicles


def detect_fiber(splitted_fascicle, weights_path, margin, threshold=0.5, crop_size=200, delta=2):
    net = Net().eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    fiber_images =[]
    boxes_s =[]

    for n, img in enumerate(splitted_fascicle):
        x = transform(img).to(device)
        y = net(x.unsqueeze(0))[0]
        boxes_ = []
        
        for i, box in enumerate(y['boxes']):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            x_d = box[2] - box[0]
            y_d = box[3] - box[1]

            # exclude boxes whose center is in the marginal zone
            if y['scores'][i] > threshold\
                and int(box[0]) > 0 and int(box[1]) > 0\
                and int(box[2]) < crop_size + 2*margin and int(box[3]) < crop_size + 2*margin\
                and margin < x_center < crop_size + margin\
                and margin < y_center < crop_size + margin\
                and x_d > 10 and y_d > 10:

                xmin = max([0, int(box[0]) - delta])
                ymin = max([0, int(box[1]) - delta])
                xmax = min([crop_size + 2*margin, int(box[2]) + delta])
                ymax = min([crop_size + 2*margin, int(box[3]) + delta])
                boxes_.append([xmin, ymin, xmax, ymax])

                fiber_image = img[ymin:ymax, xmin:xmax]
                fiber_images.append(fiber_image)
                
        boxes_s.append(boxes_)
        
    return fiber_images, boxes_s

            
def get_boxes_large(splitted_fascicle, boxes_s, crop_size, margin):
    boxes_large = []
    width = splitted_fascicle['n_w'] * crop_size
    height = splitted_fascicle['n_h'] * crop_size
    for n in range(splitted_fascicle['n_h']):
        for i in range(splitted_fascicle['n_w']):
            idx = n * splitted_fascicle['n_w'] +  i
            for box in boxes_s[idx]:
                xmin = (box[0] + crop_size * i) * width / (width + 2 * margin)
                ymin = (box[1] + crop_size * n) * height / (height + 2 * margin)
                xmax = (box[2] + crop_size * i) * width / (width + 2 * margin)
                ymax = (box[3] + crop_size * n) * height / (height + 2 * margin)
                if xmin != xmax and ymin != ymax:
                    boxes_large.append([xmin, ymin, xmax, ymax])
    return boxes_large


def visualize_detection(img_np, boxes_large, color_o, margin):
    image_PIL = Image.fromarray(img_np)
    draw = ImageDraw.Draw(image_PIL)
    for box in boxes_large:
        draw.rectangle(box, outline=color_o, width=1)
    draw.rectangle((margin, margin, img_np.shape[1]-margin, img_np.shape[0]-margin), outline=(130, 173, 194), width=3)
    return image_PIL

# def visualize_detection(img_np, boxes_large, color_o, margin, opacity=63):
#     # Convert the base color to RGBA
#     color_fill = color_o + (opacity,)

#     # Create the original image from the numpy array
#     image_PIL = Image.fromarray(img_np)

#     # Create a transparent layer
#     overlay = Image.new('RGBA', image_PIL.size, (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)

#     # Draw filled rectangles with transparency on the overlay
#     for box in boxes_large:
#         draw.rectangle(box, fill=color_fill)

#     # Draw the outer rectangle without filling on the overlay
#     draw.rectangle((margin, margin, img_np.shape[1]-margin, img_np.shape[0]-margin), outline=(130, 173, 194, 255), width=5)

#     # Blend the overlay with the original image
#     image_PIL.paste(Image.alpha_composite(image_PIL.convert('RGBA'), overlay), (0, 0))

#     return image_PIL

def get_segmentation(fiber_image, deeplab, scale, color_o, color_i, device):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    fiber_size = fiber_image.shape[:2]

    # infer
    img_PIL = Image.fromarray(fiber_image).resize((224, 224))
    img_np = np.array(img_PIL)
    img_pt = transform(img_PIL)

    y = deeplab(img_pt.unsqueeze(0).to(device))
    y_label = torch.argmax(y.cpu(), 1).squeeze(0).numpy()

    # resize to 5 times original size
    scale_5 = scale / 5
    x_PIL = Image.fromarray(fiber_image).resize([fiber_size[1]*5, fiber_size[0]*5])
    x_np = np.array(x_PIL)

    # get outer contours (y_label=2)
    outer_PIL = Image.fromarray(((y_label == 2)*255).astype(np.uint8)).resize([fiber_size[1]*5, fiber_size[0]*5])
    outer_np = np.array(outer_PIL)
    cnt_out, hier_out = cv2.findContours(outer_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnt_out) == 0:
        area_out = np.nan
        circularity = np.nan

    else:
        # get the outer contour with the largest area
        areas = [cv2.contourArea(cnt_o) for cnt_o in cnt_out]
        max_idx = np.argmax(areas)
        area_out = areas[max_idx]
        img_cnt = cv2.drawContours(x_np, [cnt_out[max_idx]], -1, color=color_o, thickness=2)
        
        # draw an ellipse
        if len(cnt_out[max_idx]) >= 5:
            perimeter = cv2.arcLength(cnt_out[max_idx], True)
            circularity = 4 * np.pi * area_out / perimeter / perimeter
            ellipse = cv2.fitEllipse(cnt_out[max_idx])
            if ellipse[1][0] >= ellipse[1][1]:
                major_axis_length = ellipse[1][0]
                minor_axis_length = ellipse[1][1]
            else:
                major_axis_length = ellipse[1][1]
                minor_axis_length = ellipse[1][0]
            eccentricity = major_axis_length / minor_axis_length
            angle = ellipse[2]
            ellipse_area = (ellipse[1][0] * ellipse[1][1]) * np.pi / 4

            hull = cv2.convexHull(cnt_out[max_idx])
            hull_perimeter = cv2.arcLength(hull, True)
            convexity = hull_perimeter / perimeter

            hull_area = cv2.contourArea(hull)
            solidity = float(area_out) / hull_area
        else:
            circularity = np.nan
            major_axis_length = np.nan
            minor_axis_length = np.nan
            angle = np.nan
            convexity = np.nan
            solidity = np.nan
            eccentricity = np.nan

        # get inner contours
        inner_PIL = Image.fromarray(((y_label == 1)*255).astype(np.uint8)).resize([fiber_size[1]*5, fiber_size[0]*5])
        inner_np = np.array(inner_PIL)
        cnt_in, hier_in = cv2.findContours(inner_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # exclude c-shape(open circle) myelin
        if len(cnt_in) !=0 and ellipse_area*0.7 < area_out < ellipse_area*1.3:
            area_in = 0
            for cnt_i in cnt_in:
                i = 0
                while True:  
                    point = [int(cnt_i[i][0][0]),int(cnt_i[i][0][1])]
                    i += 1
                    if cv2.pointPolygonTest(cnt_out[max_idx], point, False) != 1 or i == len(cnt_i):
                        break
                    if i + 1 >= len(cnt_i):
                        area = cv2.contourArea(cnt_i)
                        if area > 5:
                            img_cnt = cv2.drawContours(img_cnt, [cnt_i], -1, color=color_i, thickness=2)                                
                            area_in += area

        else:
            area_in = np.nan
            area_out = np.nan
            circularity = np.nan
            convexity = np.nan
            solidity = np.nan

    df = pd.DataFrame({
        'area_out': area_out * scale_5 * scale_5, 
        'area_in': area_in * scale_5 * scale_5,
        'perimeter': perimeter,
        'circularity': circularity,
        'convexity': convexity,
        'solidity': solidity,
        'eccentricity': eccentricity,
        'major_axis_length': major_axis_length * scale_5,
        'minor_axis_length': minor_axis_length * scale_5,
        'angle': angle,
        },index=[0])
    
    df = df.replace([0], np.nan)
    df['diameter_out'] = 2 * np.sqrt(df['area_out'] / np.pi)
    df['diameter_in'] = 2 * np.sqrt(df['area_in'] / np.pi)
    df['thickness'] = (df['diameter_out']-  df['diameter_in']) / 2
    df['g_ratio'] = df['diameter_in'] / df['diameter_out']
    segmented_fiber = {'img':np.array(x_PIL), 'img_cnt':img_cnt, 'df':df}
    return segmented_fiber


def tile_images(images, n_row=5, n_col=5):
    plt.clf()
    fig = plt.figure(figsize=(n_col*2, n_row*2))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    for n, img in enumerate(images):
        if n == n_row * n_col:
            break
        else:
            ax = fig.add_subplot(n_row, n_col, n+1)
            ax.axis('off')
            ax.imshow(img)
    fig.tight_layout()

    return fig

if __name__ == '__main__':
    main()