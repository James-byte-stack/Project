# 发送整张图片给服务器
import os
import cv2
import time
import numpy as np
import socket
import struct
import json


name = 'c004'
input_dir = f'./datasets/train/S01/{name}/frame/'
output_dir = f'./res/output_111/{name}/ref/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 图像传输 ==》编码 + 传输
# 打包的图片编码
def encode(img):
    begin = time.time()
    quality=100
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, imgencode = cv2.imencode('.jpg',img, encode_param)
    stringData = np.array(imgencode).tobytes()
    data = struct.pack('i', len(stringData)) + stringData 
    return data,round((time.time()-begin)*1000,1)

# 发送编码好的图片给server并接受检测结果/时延
def send(data,encode_time):
    # 连接服务器
    sock_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_server.connect(('192.168.3.121', 6000))
    # 发
    begin = time.time()
    sock_server.send(data)
    # 接
    # 检测结果数据大小
    results_length = struct.unpack('i', sock_server.recv(4))[0]
    results_buf = b''
    while results_length:
        # 接收图片数据
        temp_size = sock_server.recv(results_length)
        if not temp_size:
            break
        # 每次减去收到的数据大小
        results_length -= len(temp_size)
        # 将收到的数据拼接到img_data中
        results_buf += temp_size
    if results_buf == b'':
        print('results_data is empty')
        return False
    results = json.loads(results_buf.decode('utf-8'))
    time_info_length = struct.unpack('i', sock_server.recv(4))[0]
    time_info = json.loads(sock_server.recv(time_info_length).decode('utf-8')) 
    time_info = [round(e*1000,1) for e in time_info]
    transfer_time = (time.time()-begin)*1000-sum(time_info)
    time_info = [encode_time,transfer_time] + time_info
    return results,time_info
                
def inference(start_pos,end_pos=-1):
    frames = os.listdir(input_dir)
    frames = sorted(frames, key=lambda e: int(e.split(".")[0]))
    frames = frames[start_pos:end_pos]
    # 单进程
    for path in frames:
        fid = int(path.split('.')[0])
        print("Processing frame: ", fid)
        frame = cv2.imread(input_dir + path)
        data,encode_time = encode(frame)
        res,time_info = send(data,encode_time)

        with open(output_dir+'res_full.txt','a') as f:
            for d in res:
                f.write(str(fid)+' '+str(d[0]-d[2]/2)+' '+str(d[1]-d[3]/2)+' '+str(d[2])+' '+str(d[3])+' '+str(time_info)+'\n')
        with open(output_dir+'time_full.txt','a') as f:
            f.write(str(time_info)+'\n')
        with open(output_dir+'transfer_length_full.txt','a') as f:
            f.write(str(len(data))+'\n')
        


if __name__ == "__main__":
    start_pos = int(0.8*1954)
    end_pos = 1954
    begin = time.time()
    inference(start_pos,end_pos)
    print(f'Basline Time:{round((time.time() - begin)/(0.2*1954)*1000,1)} ms/frame ')
    