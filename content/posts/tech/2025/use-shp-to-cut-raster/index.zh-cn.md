---
title: '使用 shp 文件裁切（掩膜）栅格数据'
date: 2025-05-25T21:01:30+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - research skills
  - drawing
---





在地学科研绘图中，我们经常会使用 shp 文件来确定某些边界，同时又使用栅格文件（tiff, nc）来保存数据。为什么要分为两种文件呢？因为 shp 文件作为矢量数据，可以非常精确的划分边界。而栅格数据存在分辨率的问题，过粗的分辨率无法精确表示边界，过细的分辨率则会让文件变得庞大。但是各种地学数据则适合用栅格数据表达，因为其本身就因为观测计数的问题存在各种不同的分辨率，其观测数据在底层逻辑上也适合组织为栅格数据。

在很多时候我并不需要整个文件存储的所有数据，我们通常只需要其中的某一区域，此时我们就需要使用对应的区域的 shp 文件来裁切。本文主要记录两种裁切方式，**一是当某个格点全部在 shp 区域的边界内时，认为此格点在区域内；二是某个格点只要部分在边界内，则认为此格点在边界内**。我将这两种分别称为**严格裁切**和**外扩裁切**。

> 本文所指裁切主要以生成所需区域的 mask 的形式进行。

## 一、严格裁切

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from shapely.vectorized import contains

shp_fp = "your_file.shp"  # 你的shp文件路径
gdf = gpd.read_file(shp_fp) # 如果包含多个polygon也可以，gdf.geometry合并即可
# 假如gdf不是EPSG:4326，则转换
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326") 

# 此部分根据需要裁切的数据的分辨率和 .shape 确定，例如我使用 60S~90N 180W~180E 分辨率 0.25°的数据，则 shape 为 (600, 1440)。
# 以维度 -60 到 90，精度 -180 到 180，格点宽高 0.25，构建经纬度数组。
ny, nx = 600, 1440
lats = np.linspace(90 - 0.25/2, -90 + 0.25/2, ny)   # 89.875, ..., -89.875
lons = np.linspace(-180, 180 - 0.25, nx)            # -180, ..., 179.75

lon2d, lat2d = np.meshgrid(lons, lats) # 构建全球经纬度矩阵

# 获取所有Polygon合并后的对象
# 适配多Polygon：unary_union
greenland_shape = gdf.geometry.unary_union

mask = contains(greenland_shape, lon2d, lat2d) # contains 判断格点是否在图形内，注意后两个参数代表 x,y 坐标，需要使用 meshgrid 生成。

plt.imshow(mask, origin='upper', extent=[lons[0], lons[-1], lats[-1], lats[0]])
plt.title("ask")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

```

## 二、外扩裁切

```python
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
import rasterio
from rasterio.transform import from_origin

# 1. 读取Greenland多边形(shp)
gdf = gpd.read_file("your_file.shp")
if gdf.crs != 'EPSG:4326':
    gdf = gdf.to_crs("EPSG:4326")  # 保证是经纬度投影

your_shape = gdf.geometry.unary_union  # 合并成单个多边形
height, width = 600, 1440

# 用rasterio.Affine更加清晰
# transform = from_origin(-180, 90, 0.25, 0.25)  # 左上角-West,North, dlon, dlat

# 参考数据（如变量数据）形状是(height, width)
mask = rasterize(
    [(your_shape, 1)],  # 需要一个list of (geometry, value)
    out_shape=(height, width),
    transform=transform,
    fill=0,
    all_touched=True,        # 关键：只要格子有被touch就算True
    dtype='uint8'
)
mask = mask.astype(bool)
import matplotlib.pyplot as plt
plt.imshow(mask, origin='upper')
plt.title("mask")
plt.show()

```

