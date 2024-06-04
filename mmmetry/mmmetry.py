import os
# import os.path
import shutil
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
# import seaborn as sns
import cv2
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from PIL import Image
from PIL import ImageDraw
import warnings
warnings.simplefilter('ignore')
import gdown
import zipfile
from tqdm import tqdm
from ultralytics import YOLO

import networkx as nx
from community import community_louvain
import lightgbm as lgb


def main(root, wsi_path, scale):
    seg_weight = f'{root}/weights/mmmetry_0819.pt'
    lgb_weights = sorted(glob(f'{root}/weights/lgb_model_*.txt'))
    weight_paths = lgb_weights.copy()
    weight_paths.append(seg_weight)
    for path_wt in weight_paths:
        if not os.path.exists(path_wt):
            url = 'https://drive.google.com/uc?id=1KL7NbGX0K0NAnV31ZrF8vBn-JYz9e4N2'
            gdown.download(url, root, quiet=False)
            zip_path = glob(f'{root}/mmmetry*tmp')[0]
            output = f'{root}/weights'
            os.makedirs(output, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output)
            os.remove(zip_path)
            break
        

    # set parameters
    margin = 300
    crop_size = 1400
    size = (640, 640)
    
    # infer
    data = []
    image_id = os.path.basename(wsi_path).split('.')[0]
    save_dir = f'{root}/results/{image_id}' 
    os.makedirs(save_dir, exist_ok=True)

    image = Image.open(wsi_path)
    img_np = np.array(image)
    original_size = img_np.shape[:2]

    # tile image
    tiled_images = tile_image(img_np, crop_size=crop_size, margin=margin)

    # predict
    model = YOLO(seg_weight)
    results= model.predict(tiled_images['image'], save=False, save_txt=False)

    # get bboxes and masks
    threshold = 0.7
    boxes_large = []
    masks_large = []
    for n_tile, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        if boxes.shape == (0, 4):
            continue
        probs= result.boxes.conf.cpu().numpy()

        masks = result.masks.xy
        n_col = tiled_images['num_row_col'][n_tile][1]
        n_row = tiled_images['num_row_col'][n_tile][0]
        for n_box, box in enumerate(boxes):
            if probs[n_box] < threshold:
                continue
            boxes_ = [
                        max(0, box[0] + n_col * crop_size - margin),
                        max(0, box[1] + n_row * crop_size - margin),
                        min(original_size[1], box[2] + n_col * crop_size - margin),
                        min(original_size[0], box[3] + n_row * crop_size - margin),
                    ]
            boxes_large.append(boxes_)
            
            points = masks[n_box]
            masks_ = [[x + n_col * crop_size - margin, y + n_row * crop_size - margin] for x, y in points]
            masks_large.append(masks_)
            
    # Exclude overlaps
    bboxes = boxes_large
    filtered_bboxes = []
    filtered_masks = []
    area_threshold = 0.1
    for i, bbox1 in enumerate(bboxes):
        keep = True

        for j, bbox2 in enumerate(bboxes):
            if i == j:
                continue
            x1min, y1min, x1max, y1max = bbox1
            x2min, y2min, x2max, y2max = bbox2
            x_left = max(x1min, x2min)
            y_top = max(y1min, y2min)
            x_right = min(x1max, x2max)
            y_bottom = min(y1max, y2max)
            if x_right < x_left or y_bottom < y_top:
                continue

            bbox1_area = (x1max - x1min) * (y1max - y1min)
            bbox2_area = (x2max - x2min) * (y2max - y2min)
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            bbox1_only_area = bbox1_area - intersection_area
            iou = intersection_area / (bbox1_area + bbox2_area - intersection_area)

            if bbox1_area < bbox2_area:
                if bbox1_only_area / bbox2_area < area_threshold:
                    keep = False
                    break
                elif bbox1_only_area / iou < area_threshold:
                    keep = False
                    break
                elif iou > 1 - area_threshold:
                    keep = False
                    break
            if bbox1_area == bbox2_area:
                if i < j:
                    keep = False
                    break

        if keep:
            filtered_bboxes.append(bbox1)
            filtered_masks.append(masks_large[i])

    # get binary masks
    masks_bin = []
    for polygon in filtered_masks:
        polygon = np.array([(x, y) for x, y in polygon], dtype=np.int32)
        mask = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        masks_bin.append(mask)
        
        
    # calculate morphological parameters
    for i, contour in enumerate(tqdm(filtered_masks)):
        bbox = filtered_bboxes[i]
        mask_bin = masks_bin[i][bbox[1]:bbox[3], bbox[0]:bbox[2]]
        contour = np.array(contour, dtype=np.float32)
        area = cv2.contourArea(contour) * scale * scale
        perimeter = cv2.arcLength(contour, True) * scale
        circularity = 4 * np.pi * area / perimeter / perimeter * 100

        ellipse = cv2.fitEllipse(contour)
        if ellipse[1][0] >= ellipse[1][1]:
            major_axis_length = ellipse[1][0]
            minor_axis_length = ellipse[1][1]
        else:
            major_axis_length = ellipse[1][1]
            minor_axis_length = ellipse[1][0]
        eccentricity = major_axis_length / minor_axis_length
        angle = ellipse[2]


        hull = cv2.convexHull(contour)
        hull_perimeter = cv2.arcLength(hull, True) * scale
        convexity = hull_perimeter / perimeter

        hull_area = cv2.contourArea(hull) * scale * scale
        solidity = float(area) / hull_area
        os.makedirs(f'{save_dir}/masks', exist_ok=True)
        Image.fromarray(mask_bin).save(f'{save_dir}/masks/mask_{image_id}_{i}.jpg')
        dict = {
            'image_id': image_id,
            'obj_id': i,
            'bbox': bbox,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'convexity': convexity,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'angle': angle,
        }
        data.append(dict)
    df = pd.DataFrame(data)
    # df = df[df['area'] < 10000]
    df.to_csv(f'{save_dir}/{image_id}.csv')
    

    # exclude
    df_morph = pd.DataFrame([])
    df_morph = df.drop(['convexity', 'solidity', 'angle'], axis=1)
    df_morph = df_morph[df_morph['area'] < 20000]
    df_morph = df_morph[df_morph['area'] > 100]
    df_morph = df_morph[df_morph['eccentricity'] < 5 ]
    
    # compute z-score
    log_area_mean = 7.96014217026706
    log_area_SD = 0.5911355878980776
    circularity_mean = 68.86930830039762
    circularity_SD = 10.027856984522497
    area_l_th, area_h_th, circularity_l_th, circularity_h_th = (1543.7339308639448, 5211.050850521912, 58.78273192305205, 78.9064387797372)    
    df_morph['area_nl'] = (np.log(df_morph['area']) - log_area_mean) / log_area_SD
    df_morph['circularity_nl'] = (df_morph['circularity'] - circularity_mean) / circularity_SD
    
    # visualize
    r = 10
    data = df_morph.copy()
    data['bbox'] = data['bbox'].apply(lambda x: [int(int(s) / r) for s in x])

    # resize
    size = [int(int(s) / r)  for s in Image.open(wsi_path).size]
    img_np = np.array(Image.open(wsi_path).resize(size))

    wsi_h, wsi_w, _ = img_np.shape

    # save wsi
    Image.fromarray(img_np).save(f'{save_dir}/{image_id}.jpg')
 

    # get masks of area
    masks = np.zeros((wsi_h, wsi_w))
    for row in tqdm(data.itertuples()):
        path = f'{save_dir}/masks/mask_{image_id}_{row.obj_id}.jpg'
        w, h = [int(s / r) for s in Image.open(path).size]
        mask_np = np.array(Image.open(path).resize((w, h)))
        
        x, y = row.bbox[:2]

        mask_pad_np = cv2.copyMakeBorder(mask_np, y, wsi_h-h-y, x, wsi_w-w-x, 0)
        if row.area_nl == 0:
            row.area_nl = 0.0000001
        mask_bin = (mask_pad_np > 127) * row.area_nl 
        masks += mask_bin

    cmap = plt.get_cmap('viridis')
    masks[masks == 0] = False
    masks = np.clip(masks, a_min=-3, a_max=3)
    color_mask = cmap((masks + 3) / 6)[:, :, :3]
    color_mask[masks==False] = 1

    color_mask  = (color_mask * 255).astype(np.uint8)
    merged = cv2.addWeighted(img_np, 0.3, color_mask, 0.7, 0)
    Image.fromarray(merged).save(f'{save_dir}/{image_id}_area.jpg')


    # get masks of circularity
    masks = np.zeros((wsi_h, wsi_w))
    for row in tqdm(data.itertuples()):
        path = f'{save_dir}/masks/mask_{image_id}_{row.obj_id}.jpg'
        w, h = [int(s / r) for s in Image.open(path).size]
        mask_np = np.array(Image.open(path).resize((w, h)))
        
        x, y = row.bbox[:2]

        mask_pad_np = cv2.copyMakeBorder(mask_np, y, wsi_h-h-y, x, wsi_w-w-x, 0)
        if row.circularity_nl == 0:
            row.circularity_nl = 0.0000001
        mask_bin = (mask_pad_np > 127) * row.circularity_nl 
        masks += mask_bin

    cmap = plt.get_cmap('viridis')
    masks[masks == 0] = False
    masks = np.clip(masks, a_min=-3, a_max=3)
    color_mask = cmap((masks + 3) / 6)[:, :, :3]
    color_mask[masks==False] = 1

    color_mask  = (color_mask * 255).astype(np.uint8)
    merged = cv2.addWeighted(img_np, 0.3, color_mask, 0.7, 0)
    Image.fromarray(merged).save(f'{save_dir}/{image_id}_circularity.jpg')

    #  get masks of circularity in small fibers
    masks = np.zeros((wsi_h, wsi_w))
    for row in tqdm(data.itertuples()):
        path = f'{save_dir}/masks/mask_{image_id}_{row.obj_id}.jpg'
        w, h = [int(s / r) for s in Image.open(path).size]
        mask_np = np.array(Image.open(path).resize((w, h)))
        
        x, y = row.bbox[:2]

        mask_pad_np = cv2.copyMakeBorder(mask_np, y, wsi_h-h-y, x, wsi_w-w-x, 0)
        # area z-score < -1, mean - SD
        if row.area_nl > -1:
            continue
        if row.circularity_nl == 0:
            row.circularity_nl = 0.0000001
        mask_bin = (mask_pad_np > 127) * row.circularity_nl 
        masks += mask_bin

    cmap = plt.get_cmap('viridis')
    masks[masks == 0] = False
    masks = np.clip(masks, a_min=-3, a_max=3)
    color_mask = cmap((masks + 3) / 6)[:, :, :3]
    color_mask[masks==False] = 1

    color_mask  = (color_mask * 255).astype(np.uint8)
    merged = cv2.addWeighted(img_np, 0.3, color_mask, 0.7, 0)
    Image.fromarray(merged).save(f'{save_dir}/{image_id}_circularity_small_fibers.jpg')

    # get masks of small angular fibers
    masks = np.zeros((wsi_h, wsi_w))
    for row in tqdm(data.itertuples()):
        path = f'{save_dir}/masks/mask_{image_id}_{row.obj_id}.jpg'
        w, h = [int(s / r) for s in Image.open(path).size]
        mask_np = np.array(Image.open(path).resize((w, h)))
        
        x, y = row.bbox[:2]

        mask_pad_np = cv2.copyMakeBorder(mask_np, y, wsi_h-h-y, x, wsi_w-w-x, 0)

        # roundness z-score < -1
        if row.circularity_nl > -1:
            continue
        # area z-score < -1, mean - SD
        if row.area_nl > -1:
            continue
        if row.area_nl == 0:
            row.area_nl = 0.0000001
        mask_bin = (mask_pad_np > 127) * -2 # why -2?
        masks += mask_bin

    cmap = plt.get_cmap('viridis')
    masks[masks == 0] = False
    masks = np.clip(masks, a_min=-3, a_max=3)
    color_mask = cmap((masks + 3) / 6)[:, :, :3]
    color_mask[masks==False] = 1

    color_mask  = (color_mask * 255).astype(np.uint8)
    merged = cv2.addWeighted(img_np, 0.3, color_mask, 0.7, 0)
    Image.fromarray(merged).save(f'{save_dir}/{image_id}_small_angular_fibers.jpg')


    # get masks of small round fibers
    masks = np.zeros((wsi_h, wsi_w))
    for row in tqdm(data.itertuples()):
        path = f'{save_dir}/masks/mask_{image_id}_{row.obj_id}.jpg'
        w, h = [int(s / r) for s in Image.open(path).size]
        mask_np = np.array(Image.open(path).resize((w, h)))
        
        x, y = row.bbox[:2]

        mask_pad_np = cv2.copyMakeBorder(mask_np, y, wsi_h-h-y, x, wsi_w-w-x, 0)

        # circularity z-score > 1, mean + SD
        if row.circularity_nl < 1:
            continue
        # area z-score < -1, mean - SD
        if row.area_nl > -1:
            continue
        if row.area_nl == 0:
            row.area_nl = 0.0000001
        mask_bin = (mask_pad_np > 127) * 2 # why -2?
        masks += mask_bin

    cmap = plt.get_cmap('viridis')
    masks[masks == 0] = False
    masks = np.clip(masks, a_min=-3, a_max=3)
    color_mask = cmap((masks + 3) / 6)[:, :, :3]
    color_mask[masks==False] = 1

    color_mask  = (color_mask * 255).astype(np.uint8)
    merged = cv2.addWeighted(img_np, 0.3, color_mask, 0.7, 0)
    Image.fromarray(merged).save(f'{save_dir}/{image_id}_small_round_fibers.jpg')

        
    df_case = df_morph.loc[:, ['image_id', 'area', 'perimeter', 'circularity', 'eccentricity']].groupby(['image_id']).agg(['mean', 'std', 'count'])
    df_case.columns = [
        'area_mean',
        'area_std',
        'count',
        'perimeter_mean',
        'perimeter_std',
        'perimeter_count',
        'circularity_mean',
        'circularity_std',
        'circularity_count',
        'eccentricity_mean',
        'eccentricity_std',
        'eccentricity_count',

        ]
    df_case = df_case.reset_index()
    df_morph[['x', 'y']] = df_morph['bbox'].apply(calculate_center).apply(pd.Series)

    # df_graph = pd.DataFrame({'image_id': [image_id]})
    min_nodes = 3
    radius = 50
    points = df_morph[(df_morph['image_id'] == image_id) & (df_morph['area'] < area_l_th)][['x', 'y']]
    points *= scale  # Scale points [µm]

    if len(points) < 10:
        df_case.loc[df_case['image_id']==image_id, f'avg_nodes'] = 0
        df_case.loc[df_case['image_id'] == image_id, f'avg_edges_per_node'] = 0
        df_case.loc[df_case['image_id'] == image_id, f'avg_edges_per_node'] = 0
        
    else:   
        # Initialize an empty graph
        G = nx.Graph()
        # Adding nodes with specific positions
        for i, point in enumerate(points.values):
            G.add_node(i, pos=tuple(point))

        # Adding edges based on distance threshold
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points.values[i] - points.values[j]) <= radius:
                    G.add_edge(i, j)


        # Compute the best partition and create a color map
        partition = community_louvain.best_partition(G)
        filtered_nodes = [node for node, comm in partition.items() if list(partition.values()).count(comm) >= min_nodes]

        # Update the graph to only include filtered nodes
        G_filtered = G.subgraph(filtered_nodes).copy()
        partition = community_louvain.best_partition(G_filtered)
        color_map = {node: idx for idx, node in enumerate(set(partition.values()))}
        colors = [color_map[partition[node]] for node in G_filtered.nodes()]

        mag = 1000
        size = [int(int(s) / mag)  for s in Image.open(wsi_path).size]

        # Draw the graph using the positions
        pos = nx.get_node_attributes(G_filtered, 'pos')
        plt.figure(figsize=size)  
        nx.draw_networkx_nodes(G_filtered, pos, node_size=20, cmap='prism', node_color=colors, label=partition)
        nx.draw_networkx_edges(G_filtered, pos, alpha=1)
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.savefig(f'{save_dir}/{image_id}_graph.jpg') 

        unique_clusters = set(partition.values())
        num_clusters = len(unique_clusters)
        print(image_id)
        # print(f"Number of clusters: {num_clusters}")
        if num_clusters == 0:
            df_case.loc[df_case['image_id']==image_id, f'avg_nodes'] = 0
            df_case.loc[df_case['image_id'] == image_id, f'avg_edges_per_node'] = 0
            df_case.loc[df_case['image_id'] == image_id, f'avg_edges_per_node'] = 0

        else:
            # Calculate the number of nodes in each cluster
            cluster_sizes = {cluster: 0 for cluster in unique_clusters}


            for node in partition:
                cluster_sizes[partition[node]] += 1

            # Calculate average nodes per cluster
            average_nodes_per_cluster = sum(cluster_sizes.values()) / num_clusters
            # print(f"Average number of nodes per cluster: {average_nodes_per_cluster:.2f}")
            avg_edges_per_node = np.mean([deg for node, deg in G_filtered.degree()])
            # print(f"Average number of edges per node: {avg_edges_per_node:.2f}")

            df_case.loc[df_case['image_id']==image_id, f'n_clusters'] = num_clusters
            df_case.loc[df_case['image_id']==image_id, f'avg_nodes'] = average_nodes_per_cluster
            df_case.loc[df_case['image_id'] == image_id, f'avg_edges_per_node'] = avg_edges_per_node



    circularity_in_small = df_morph[df_morph['area'] < area_l_th]['circularity'].mean()
    circularity_in_large = df_morph[df_morph['area'] > area_h_th]['circularity'].mean()
    area_h_percent = len(df_morph[df_morph['area'] > area_h_th]) / len(df_morph) * 100
    area_l_percent = len(df_morph[df_morph['area'] < area_l_th]) / len(df_morph) * 100
    round = len(df_morph[df_morph['circularity'] > circularity_h_th]) / len(df_morph) * 100
    angular = len(df_morph[df_morph['circularity'] < circularity_l_th]) / len(df_morph) * 100
    large_round = len(df_morph[(df_morph['area'] > area_h_th) & (df_morph['circularity'] > circularity_h_th)]) / len(df_morph) * 100
    large_angular = len(df_morph[(df_morph['area'] > area_h_th) & (df_morph['circularity'] < circularity_l_th)]) / len(df_morph) * 100
    small_round = len(df_morph[(df_morph['area'] < area_l_th) & (df_morph['circularity'] > circularity_h_th)]) / len(df_morph) * 100
    small_angular = len(df_morph[(df_morph['area'] < area_l_th) & (df_morph['circularity'] < circularity_l_th)]) / len(df_morph) * 100

    data_population = []
    data_population.append({
        'image_id': image_id,
        'circularity_in_small': circularity_in_small,
        'circularity_in_large': circularity_in_large,
        'large': area_h_percent,
        'small': area_l_percent,
        'round': round,
        'angular': angular,
        'large_round': large_round,
        'large_angular': large_angular,
        'small_round': small_round,
        'small_angular': small_angular,
    })

    df_case = pd.merge(df_case, pd.DataFrame(data_population),  how='inner', on='image_id')
    df_case['grouped_fibers'] = (df_case['avg_nodes'] * df_case['n_clusters']) / (df_case['small'] * df_case['count']) * 10000

    cols = [
        'area_mean', 
        'area_std', 
        'large', 'small',
        'circularity_mean', 
        'circularity_std',
        'circularity_in_small', 'circularity_in_large', 
        'round_circ', 'angular_circ', 
        'large_round_circ', 'large_angular_circ', 
        'small_round_circ', 'small_angular_circ',
        'grouped_small_fibers',
        'small_avg_nodes', 
        'small_avg_edges_per_node', 
        ]

    df_case.rename(columns={
        'round': 'round_circ', 
        'angular': 'angular_circ', 
        'large_round': 'large_round_circ', 
        'large_angular': 'large_angular_circ',
        'small_round': 'small_round_circ', 
        'small_angular': 'small_angular_circ', 
        'grouped_fibers': 'grouped_small_fibers',
        'avg_nodes': 'small_avg_nodes',
        'avg_edges_per_node': 'small_avg_edges_per_node', 
    }, inplace=True)
    df_case.to_csv(f'{save_dir}/summary.csv')
    df_lgb = df_case[cols]

    # Loading the model
    lgb_weights = sorted(glob(f'{root}/weights/lgb_model_*.txt'))
    labels = ['Normal biopsy', 'Myopathy', 'Neuropathy']
    predictions = pd.DataFrame(columns=labels)
    for lgb_weight in lgb_weights:
        loaded_model = lgb.Booster(model_file=lgb_weight)
        prediction = loaded_model.predict(df_lgb)
        pred_df = pd.DataFrame(prediction, columns=labels)
        predictions = pd.concat([predictions, pred_df], ignore_index=True)
    print('Prediction by LightGBM model')
    for idx, row in predictions.agg(['mean', 'std']).T.iterrows():
        print(f"  {row.name}: {row['mean']*100:.03f}±{row['std']*100:.03f}%")

    shutil.rmtree(f'{save_dir}/masks')
    return df_case

def tile_image(img_np, crop_size, margin):
    fas_tile = []
    width = img_np.shape[1]
    height = img_np.shape[0]
    n_col = math.ceil((width + 2 * margin) / crop_size)
    n_row = math.ceil((height + 2 * margin) / crop_size)
    # padding
    pad_w = crop_size * n_col - width - margin
    pad_h = crop_size * n_row - height - margin
    img_np = cv2.copyMakeBorder(img_np, margin, pad_h, margin, pad_w, cv2.BORDER_CONSTANT, (0, 0, 0))
    num_row_col = []

    for row in range(n_row):
        for col in range(n_col):
            fas_tile.append(img_np[row*crop_size:(row+1)*crop_size+2*margin, col*crop_size:(col+1)*crop_size+2*margin])
            num_row_col.append((row, col))
    tile_dict = {'image': fas_tile, 'num_row_col': num_row_col, 'original_size': (width, height)}

    return tile_dict

def visualize_detection(img_np, boxes_large):
    image_PIL = Image.fromarray(img_np)
    draw = ImageDraw.Draw(image_PIL)
    for box in boxes_large:
        draw.rectangle(box, outline='blue', width=2)
    # draw.rectangle((margin, margin, img_np.shape[1]-margin, img_np.shape[0]-margin), outline='green', width=3)
    return image_PIL

def calculate_center(box):
    x_min, y_min, x_max, y_max = box
    
    # scale [px]
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center, y_center

if __name__ == '__main__':
    main()