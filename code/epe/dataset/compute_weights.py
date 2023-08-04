from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from epe.matching import load_matching_crops

# 添加一些命令：匹配的patch的路径、图像高度、宽度、权重文件路径
p = ArgumentParser('Compute sampling weights for matches across datasets. \
	These weights ensure that we sample spatially uniformly from the source dataset \
	even if we have a skewed distribution of matches across datasets.')
p.add_argument('matched_crop_path', type=Path, help="Path to csv with matched crop info.")
p.add_argument('height', type=int, help="Height of images in dataset.")
p.add_argument('width', type=int, help="Width of images in dataset.")
p.add_argument('weight_path', type=Path, help="Path to (output) weight file.")
args = p.parse_args()
# load_matching_crops():从csv中取出匹配的两个图片的坐标以及他们各种的crop坐标->tuple(list,list)
# 注意src_crops=[(src_path,r0,r1,c0,c1),(),(),......,()]
src_crops,_ = load_matching_crops(args.matched_crop_path)

d = np.zeros((args.height, args.width), dtype=np.int32) 
print('Computing density...')

for s in tqdm(src_crops): # 将crop区域内所有像素值变+1.
	d[s[1]:s[2],s[3]:s[4]] += 1 

print('Computing individual weights...')

w = np.zeros((len(src_crops), 1)) # 对每张图生成一个权重
for i, s in enumerate(tqdm(src_crops)):
	w[i,0] = np.mean(d[s[1]:s[2],s[3]:s[4]]) # 这个w等于这张图的mask的所有数值取均值
	pass

N = np.max(d)
p = N / w
np.savez_compressed(args.weight_path, w=p)