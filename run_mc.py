# device端做的事情：
# 1. ROI提取
# 3. 利用文件数据判断视野重叠区的ROI需不需要进行后续处理
# 4. 打包
# 5. ROI数据发送到server[图像+打包信息/map_info+map_camera[参考相机是哪个]]
# 6. 数据接受


# c002
from rectpack import newPacker
import cv2
import numpy as np
import time
import os
import struct
import socket
import json
from joblib import load
import pandas as pd
from multiprocessing import Pool,Lock


import warnings

from sort import Sort


warnings.filterwarnings("ignore")

expansion_scale = 0.3
downSample_threshold = 160 * 160

bin_w = 1920
bin_h = 320

backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

link_threshold = 0.2
overlap_threshold = 0.8
train_ratio = 0.8

name = 'c004'
ref = 'c002'
input_dir = f'./datasets/train/S01/{name}/frame/'
output_dir = f'./res/output_4/{name}/'


# 红绿灯
singles = {'c001':[[1613, 139, 57, 139],[976, 14, 664, 125]],
           'c002':[[1839.5, 230.0, 51, 106],[192.0, 138.0, 78, 50],[949.0, 72.0, 116, 62],[1462.5, 48.0, 39, 96],[1266.0, 54.5, 54, 109],[556.0+54, 21.0+53/2, 119, 53]],
           'c003':[[1008, 0, 356, 58]],
           'c004':[[389.0, 258.5, 60, 79],[194.0, 210.5, 248, 125],[1249.5, 110.5, 439, 113]]}

# 重叠区域
overlap_path = f'./choose_ref/{ref}_{name}/blocks_{name}_merge.txt'
overlap = []
with open(overlap_path, 'r') as f:
    for line in f.readlines():
        block = eval(line.strip())
        overlap.append(block)

df = pd.read_csv(f'./roi_info/{name}_rois_info.txt', sep=' ', header=None)

def iou(b1, b2):
    """
    calculate intersection over union
    """
    (x1, y1, w1, h1) = b1[:4]
    (x2, y2, w2, h2) = b2[:4]
    x3 = max(x1, x2)
    y3 = max(y1, y2)
    x4 = min(x1+w1, x2+w2)
    y4 = min(y1+h1, y2+h2)
    if x3 > x4 or y3 > y4:
        return 0,0
    else:
        overlap = abs(x4-x3)*abs(y4-y3)
        b1_area = w1 * h1
        b2_area = w2 * h2
        return overlap, overlap/(b1_area+b2_area-overlap)

def iou_merge(b1, b2):
    """
    calculate intersection over union
    """
    overlap_area = iou(b1,b2)[0]
    w1,h1 = b1[2],b1[3]
    w2,h2 = b2[2],b2[3]
    b1_area = w1 * h1
    b2_area = w2 * h2
    return overlap_area / min((b1_area + b2_area - overlap_area),b1_area,b2_area)

# 判断两个bbox是否可以合并
# bbox1: x,y,w,h,s,x_prev,y_prev
# bbox2: x,y,w,h,s,n,C
def can_merge(bbox1,bbox2):
    bbox1 = [bbox1[0]-bbox1[2]/2,bbox1[1]-bbox1[3]/2,bbox1[2],bbox1[3]]
    bbox2 = [bbox2[0]-bbox2[2]/2,bbox2[1]-bbox2[3]/2,bbox2[2],bbox2[3]]
    if iou_merge(bbox1,bbox2)  > 0.7:
        return True
    x = min(bbox1[0],bbox2[0])
    y = min(bbox1[1],bbox2[1])
    w = max(bbox1[0]+bbox1[2],bbox2[0]+bbox2[2]) - x
    h = max(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3]) - y
    # x轴上连接，y轴上重叠
    if max(bbox2[0],bbox1[0]) - min(bbox2[0]+bbox2[2],bbox1[0]+bbox1[2]) < link_threshold * w and  min(bbox2[1]+bbox2[3],bbox1[1]+bbox1[3]) - max(bbox2[1],bbox1[1]) > overlap_threshold * h:
        return True
    # y轴上连接，x轴上重叠
    if max(bbox2[1],bbox1[1]) - min(bbox2[1]+bbox2[3],bbox1[1]+bbox1[3]) < link_threshold * h and  min(bbox2[0]+bbox2[2],bbox1[0]+bbox1[2]) - max(bbox2[0],bbox1[0]) > overlap_threshold * w:
        return True
    return False

# 判断bbox是否在overlap区域
# overlap: 一系列bbox
# bbox：x,y,w,h
def is_in_overlap(roi):
    roi_area = roi[2] * roi[3]
    area_sum = 0
    for bbox in overlap:
        area_sum += iou(roi, bbox)[0]
    if area_sum/roi_area < 0.7:
        return False
    else:
        return True


def findRoisByBbox():
    """
    find rois by bbox from previous result
    """
    res_file = os.path.join(output_dir, 'res.txt')
    if (not os.path.isfile(res_file)) or (os.path.getsize(res_file) == 0):
        # 文件不存在或是空文件 => 无上一帧roi
        return []

    # rois_ignore:红绿灯，格式x_c,y_c,w,h
    rois_ignore = singles[name]
    rois = []
    # 用文件的最后一行作为prev_res
    df = pd.read_csv(res_file,sep=' ',header=None)
    prev_res = df[df[0] == df[0].max()].values.tolist()
    for roi in prev_res:
        x,y,w,h,deta_x,deta_y = roi[1:7]
        x_curr = x + deta_x
        y_curr = y + deta_y
        w_curr = w * (1+expansion_scale)  # 膨胀系数可能带来超边界的问题
        h_curr = h * (1+expansion_scale)
        flag = 1
        for roi in rois_ignore:
            x_r,y_r,w_r,h_r = roi[:4]
            if iou_merge([x-w/2,y-h/2,w,h],[x_r-w_r/2,y_r-h_r/2,w_r,h_r]) > 0.7:
                flag = 0
                break
        if flag:
            scale = min(min(1, downSample_threshold/(w_curr*h_curr)),min(1,bin_w/w_curr),min(1,bin_h/h_curr))
            # 超界限处理
            if x_curr - w_curr/2 < 0: x_curr = w_curr/2
            if y_curr - h_curr/2 < 0: y_curr = h_curr/2
            if x_curr + w_curr/2 > 1920: x_curr = 1920 - w_curr/2
            if y_curr + h_curr/2 > 1080: y_curr = 1080 - h_curr/2
            rois.append([x_curr,y_curr,w_curr,h_curr,scale,x,y])
    return rois


# 只在uniquePatch之外范围里面操作
def findRoisOfNewObject(img,rois_ignore):

    cols = 640
    rows = 360
    img_new = cv2.resize(img,(cols, rows))
    rois_new = []

    fgMask = backSub.apply(img_new)

    th = cv2.threshold(fgMask.copy(), 240, 255, cv2.THRESH_BINARY)[1]
    # 腐蚀
    eroded = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # 膨胀
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    dilated = cv2.erode(dilated, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if  len(contours) < 1: # 背景没建好
        return 'shaking'
    elif len(contours) > 30: # 背景建好了，但是有很多噪点
        # 只选大于0.0005*1920*1080的部分
        contours = [c for c in contours if cv2.contourArea(c) > 0.0005*1920*1080]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        x = x * 1920 / cols
        y = y * (1080) / rows
        w = w * 1920 / cols
        h = h * (1080) / rows
        # 去掉已经选了的部分
        flag = 1
        for roi in rois_ignore:
            x_r,y_r,w_r,h_r = roi[:4]
            if iou_merge([x,y,w,h],[x_r-w_r/2,y_r-h_r/2,w_r,h_r]) > 0.7:
                flag = 0
                break
        if flag:
            scale = min(1, downSample_threshold/(w*h))
            rois_new.append([x+w/2,y+h/2,w,h,scale,x+w/2,y+h/2])
    return rois_new


# integrate the adjacent rois by depth-first search algorithm
def merge(patches):
    stack = patches
    result = []
    while stack:
        current_patch = stack.pop()
        flag = False
        for i, patch in enumerate(result):
            if can_merge(current_patch, patch):
                # 合并
                x = min(current_patch[0]-current_patch[2]/2,patch[0]-patch[2]/2)
                y = min(current_patch[1]-current_patch[3]/2,patch[1]-patch[3]/2)
                w = max(current_patch[0]+current_patch[2]/2,patch[0]+patch[2]/2) - x
                h = max(current_patch[1]+current_patch[3]/2,patch[1]+patch[3]/2) - y
                result[i] = [x+w/2,y+h/2,w,h,max(patch[4],current_patch[4]),patch[-2]+current_patch[-2],patch[-1]+current_patch[-1]]
                flag = True
                break
        if not flag:
            result.append(current_patch)
    return result



def addNumber(rois,flag_shaking=False):
    """
    添加每个patch的潜在物体数量并进行合并:格式[x_c,y_c,w,h,scale,num,{num个(x_b,y_b)}]
    """
    # rois: [x_c,y_c,w,h,scale,x_prev,y_prev]
    if flag_shaking:
        return []
    key_regions = []
    # regions每个元素取小数点后三位
    regions_dect = [[round(e,3) for e in r]for r in rois]
    regions_process = [None for _ in range(len(regions_dect))]
    for idx,region in enumerate(regions_dect):
        n = 3
        regions_process[idx] = region[:-2] + [n,[[round(region[-2],3),round(region[-1],3)] for _ in range(n)]]
    # regions_process按照面积排序，大的在前面
    regions_process = sorted(regions_process, key=lambda e: (e[2]*e[3]), reverse=False)
    key_regions = merge(merge(regions_process))
    return key_regions 

# 将一些列bbox打包成一张图，记录后续进行remap
def packing(patches,img):
    if len(patches) == 0: # 这里可以加上是否是参考相机的判断，只传uniquePatch
        new_img = img
        map_info = None
        return new_img,map_info
    # patch: [x_c,y_c,w,h,scale,num,{num个(x_b,y_b)}] 
    # 按照面积给patches排序
    patches = sorted(patches, key=lambda e: (e[2]*e[3]), reverse=True)
    patches = [[round(e,3) for e in kr[:-2]]+kr[-2:] for kr in patches]
    # patches处理：缩放 + int + 超界限处理
    patches_process = [None for _ in range(len(patches))]
    patches_scale = [None for _ in range(len(patches))]
    for idx,patch in enumerate(patches):
        x,y,w,h,scale = patch[:5]
        x,y,w,h = int((x-w/2)),int((y-h/2)),int(w),int(h)
        x = max(0,x)
        y = max(0,y)
        w = min(w,1920-x)
        h = min(h,1080-y)
        patches_process[idx] = [x,y,w,h]
        patches_scale[idx] = [int(x*scale),int(y*scale),int(w*scale),int(h*scale)]
    rectangles = [(patch[2],patch[3])for patch in patches_scale]
    bins = [(bin_w, bin_h),(bin_w, bin_h),(bin_w, bin_h)]
    packer = newPacker(rotation=False)
    map_info = [{} for _ in range(len(bins))]
    for id,patch in enumerate(rectangles):
        packer.add_rect(*patch,id)
    for b in bins:
        packer.add_bin(*b)
    packer.pack()
    all_rects = packer.rect_list()
    new_imgs = [np.zeros((bin_h,bin_w,3),dtype=np.uint8) for _ in range(len(bins))]
    new_xs = [0 for _ in range(len(bins))]
    new_ys = [0 for _ in range(len(bins))]
    for rect in all_rects:
        b, x, y, w, h, rid = rect
        new_imgs[b][int(y):int(y)+int(h), int(x):int(x)+int(w)] = cv2.resize(img[int(patches_process[rid][1]):int(patches_process[rid][1])+int(patches_process[rid][3]),int(patches_process[rid][0]):int(patches_process[rid][0])+int(patches_process[rid][2])],
                                                                    (int(w),int(h)))
        map_info[b][f"({x},{y},{w},{h})"] = patches[rid]
        new_xs[b] = int(max(new_xs[b],x+w))
        new_ys[b] = int(max(new_ys[b],y+h))
    new_imgs = [new_imgs[b][:new_ys[b],:new_xs[b],:] for b in range(len(bins)) if new_ys[b] and new_xs[b]]
    if len(new_imgs) == 1:
        return new_imgs[0],map_info[0]
    elif len(new_imgs) == 2: # 把两张图拼成一张
        combine_img = np.zeros((new_ys[0]+new_ys[1],max(new_xs),3),dtype=np.uint8)
        combine_mapinfo = {}
        # combine_img[:max(new_ys),:,:] = new_imgs[0][:max(new_ys),:,:]
        # combine_img[max(new_ys):max(new_ys)+new_ys[1],:,:] = new_imgs[1][:new_ys[1],:,:]
        combine_img[:new_ys[0],:new_xs[0],:] = new_imgs[0]
        combine_img[new_ys[0]:new_ys[0]+new_ys[1],:new_xs[1],:] = new_imgs[1]
        for idx,mi in enumerate(map_info):
            for k in mi:
                x,y,w,h = eval(k)
                combine_mapinfo[f"({x},{y+idx*new_ys[0]},{w},{h})"] = mi[k]
        return combine_img,combine_mapinfo
    else:
        return img,None

def findPatch(img,f_id):
    """
    find patch by bbox from previous result
    """
    rois = findRoisByBbox()
    rois_new = findRoisOfNewObject(img,rois+singles[name])
    rois_select,ref_name = decidePatch_2(rois,rois_new,f_id)
    key_regions = addNumber(rois_select,rois_new == 'shaking')
    return key_regions,ref_name

def sendAndrecv_Device(f_id):
    info = df[df[0] == f_id].iloc[0,1:].values
    num_ROI,area_ROI = info[0],info[1]
    return num_ROI,area_ROI

def decidePatch(rois,rois_new,f_id):
    if rois_new == 'shaking':
        return [],name
    # rois格式：[x_c,y_c,w,h,scale,x_prev,y_prev]
    # 1.摘出视野重叠区的roi
    # 2.二分模型判断需要进行后续的处理的roi
    # 3.发送和接受其他相机的roi信息[num_ROI,area_ROI]
    # 4.返回需要处理的roi
    
    # 1.摘出视野重叠区的roi
    patches = [r for r in rois + rois_new if r[2]*r[3] > 0.005*1920*1080]
    rois_OFoVs = [roi for roi in patches if is_in_overlap([roi[0]-roi[2]/2,roi[1]-roi[3]/2,roi[2],roi[3]])]
   
    # 2.二分模型判断需要进行后续的处理的roi
    if len(rois_OFoVs) == 0:
        return patches,name
    else:
        num_ROI = len(rois_OFoVs)
        area_ROI = sum([roi[2]*roi[3] for roi in rois_OFoVs])
        # 3.发送和接受其他相机的roi信息[num_ROI,area_ROI]
        num_ROI_recv,area_ROI_recv = sendAndrecv_Device(f_id)

        # 4.返回需要处理的roi
        if num_ROI < num_ROI_recv:
            res = list(filter(lambda x: x not in rois_OFoVs, patches))
            return res,ref
        elif num_ROI == num_ROI_recv and area_ROI < area_ROI_recv:
            res = list(filter(lambda x: x not in rois_OFoVs, patches))
            return res,ref
        else:
            return patches,name


def decidePatch_2(rois,rois_new,f_id):
    # 1.摘出视野重叠区的roi
    if rois_new == 'shaking':
       rois_new = []
    patches = [r for r in rois + rois_new if r[2]*r[3] > 0.005*1920*1080]
    rois_OFoVs = [roi for roi in patches if is_in_overlap([roi[0]-roi[2]/2,roi[1]-roi[3]/2,roi[2],roi[3]])]
    res = list(filter(lambda x: x not in rois_OFoVs, patches))
    return res,ref



# 图像传输 ==》编码 + 传输
# 打包的图片编码
def encode(img):
    start = time.time()
    quality = 100
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, imgencode = cv2.imencode('.jpg', img, encode_param)
    data = np.array(imgencode)
    stringData = data.tobytes()
    encode_time = time.time() - start
    return stringData, encode_time


# 发送编码好的图片给server并接受检测结果/时延
def sendAndrecv_Server(img, remap_info, map_info):
    """
    发送给服务器并接收检测结果的函数。
    :param img: 这里必须是 numpy 图像 (BGR), 由cv2.imread得到;
                函数内部再调用encode(...)获得字节流。
    :param remap_info: ROI打包映射信息,可为None
    :param map_info: 其他附加信息,可为None
    :return:
      results (list): 服务器返回的检测结果
      final_time_info (list): 包含 [encode_time_local] + 服务器端时间等
      data_len (int): 发送的数据总字节数
    """

    # 1) 建立 socket
    sock_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_server.connect(('192.168.3.121', 6000))

    # 2) 本地计时起点(若需要测量“从进入此函数到发送完”时间)
    begin = time.time()

    # 3) 调用 encode(img) => 返回 (stringData, encode_time_local)
    data_img, encode_time_local = encode(img)

    # 4) 序列化 remap_info、map_info
    data_dct = bytes(json.dumps(remap_info).encode('utf-8'))
    data_lst = bytes(json.dumps(map_info).encode('utf-8'))

    # 5) 组装要发送的数据: [4字节 data_img长度] + data_img + [4字节 data_dct长度] + data_dct + ...
    data = (
        struct.pack('i', len(data_img)) + data_img
        + struct.pack('i', len(data_dct)) + data_dct
        + struct.pack('i', len(data_lst)) + data_lst
    )

    # 这里可选：统计“打包耗时”，或 simply
    local_pack_time = time.time() - begin

    # 6) 发送
    sock_server.send(data)
    data_len = len(data)

    # 7) 接收服务器结果
    #   a) 先收 4字节 => results_length
    results_length = struct.unpack('i', sock_server.recv(4))[0]
    results_buf = b''
    while results_length:
        temp_size = sock_server.recv(results_length)
        if not temp_size:
            break
        results_length -= len(temp_size)
        results_buf += temp_size

    if results_buf == b'':
        print('results_data is empty')
        # 可以返回空或False
        sock_server.close()
        return [], [encode_time_local], data_len

    results = json.loads(results_buf.decode('utf-8'))

    #   b) 再收 4字节 => time_info_length, 然后再收 time_info
    time_info_length = struct.unpack('i', sock_server.recv(4))[0]
    time_info = json.loads(sock_server.recv(time_info_length).decode('utf-8'))

    sock_server.close()

    # 8) 组合本地encode_time 与 服务器time_info
    # 若time_info本身是 [decode_time, infer_time, postprocess_time] 等(单位秒),
    # 你可将encode_time_local加在开头, 并都换成毫秒:
    final_time_info = [encode_time_local*1000] + [t*1000 for t in time_info]

    return results, final_time_info, data_len

def inference_frame(path, fullframe=True):
    """
    :param path: 图像文件名(含扩展名)
    :param fullframe: 若为 True，则直接发送整帧; False 则只发送ROI(原逻辑).
    """
    begin = time.time()
    f_id = int(path.split(".")[0])
    img = cv2.imread(os.path.join(input_dir, path))
    print("Processing frame:", f_id)
    time_load = time.time() - begin  # 读图时间

    if fullframe:
        # =========== 整帧模式 ===========
        remap_info = None
        map_info = None
        time_findPatch = 0.0  # 整帧模式下找ROI耗时为0
    else:
        # =========== 只传ROI模式 (原逻辑) ===========
        begin_roi = time.time()
        rois_dect, ref_name = findPatch(img, f_id)
        new_img, remap_info = packing(rois_dect, img)
        map_info = [ref_name, name, f_id]
        time_findPatch = time.time() - begin_roi  # ROI查找 & packing 耗时

        img = new_img

    # =========== 发送 + 接收 ===========
    begin_send = time.time()
    # sendAndrecv_Server(...) 应该返回: (res, time_info, data_len)
    # 其中 time_info 可能是 [encode_time, decode_time, infer_time, ...] 结构
    res, time_info, data_len = sendAndrecv_Server(img, remap_info, map_info)
    # 计算纯传输时间(假设 time_info[1:] 是服务器端的分段耗时之和, 具体视你的实现)
    transfer_time = (time.time() - begin_send)*1000 - sum(time_info[1:])  # 传输时间(大概)

    # 整理 time_info, 先加上 [time_load, time_findPatch, transfer_time]
    # 再把服务器返回的 time_info(单位可能是秒)转成毫秒
    final_time_info = [time_load*1000, time_findPatch*1000, transfer_time] + time_info

    # =========== 保存结果 ===========
    # res: 检测框, 例如 [cx, cy, w, h, offsetX, offsetY]
    # 也可只保留前4个值, 视你的系统格式而定
    if fullframe:
        # 为避免与原ROI模式文件冲突, 用单独文件后缀
        with open(os.path.join(output_dir, "res_fullframe.txt"), "a") as f:
            for r in res:
                f.write(f"{f_id} {r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]}\n")

        with open(os.path.join(output_dir, "time_fullframe.txt"), "a") as f:
            f.write(str(final_time_info) + "\n")

        with open(os.path.join(output_dir, "transfer_length_fullframe.txt"), "a") as f:
            f.write(str(data_len) + "\n")
    else:
        # ROI模式下, 写原先的 res.txt / time.txt / transfer_length.txt
        with open(os.path.join(output_dir, "res.txt"), "a") as f:
            for r in res:
                f.write(f"{f_id} {r[0]} {r[1]} {r[2]} {r[3]} {r[4]} {r[5]}\n")

        with open(os.path.join(output_dir, "time.txt"), "a") as f:
            f.write(str(final_time_info) + "\n")

        with open(os.path.join(output_dir, "transfer_length.txt"), "a") as f:
            f.write(str(data_len) + "\n")


def inference(start_pos, end_pos=-1, fullframe=True):
    """
    统一处理 [start_pos, end_pos) 范围内的帧文件，
    并在每帧调用 inference_frame(path, fullframe=...)。

    :param start_pos: 起始帧索引（根据文件名排序）
    :param end_pos:   结束帧索引（不包含），-1 表示一直到最后
    :param fullframe: 若为 True，则对所有帧都用“整帧传输”；False 则用“只传ROI”模式。
    """

    # 1) 获取并排序所有帧文件
    frames = os.listdir(input_dir)
    # 按文件名中的数字排序，比如 '001.jpg' -> int('001')=1
    frames = sorted(frames, key=lambda e: int(e.split(".")[0]))
    # 截取指定范围
    frames = frames[start_pos:end_pos]

    # 2) 若输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3) 逐帧调用 inference_frame(...)
    for path in frames:
        # 在这里决定是否整帧 or ROI
        # 该逻辑由 fullframe 参数控制
        inference_frame(path, fullframe=fullframe)



if __name__ == "__main__":
    start_pos = int(0.8*1954)
    end_pos = 1954
    begin = time.time()
    inference(start_pos,end_pos,fullframe=True)
    print(f'Basline Time:{round((time.time() - begin)/(0.2*1954)*1000,1)} ms/frame ')
