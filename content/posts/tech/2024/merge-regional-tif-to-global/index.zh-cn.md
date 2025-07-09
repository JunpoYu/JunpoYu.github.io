---
title: '将多个区域的 GeoTIFF 文件合并为全球 TIFF'
date: 2024-09-14T20:50:10+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - research skills
  - gdal
---

## 一、使用 ArcMap 合并

使用 ArcMap 的工具箱对多个 TIFF 文件进行合并。

在 `ArcToolBox -> 数据管理工具 -> 栅格 -> 栅格数据集 -> 镶嵌至新栅格` 。

打开工具后，在 `输入栅格` 栏选中需要合并的数据，在 `输出位置` 选定输出路径。在`具有扩展名的栅格数据集名称` 设置输出文件名，要带扩展名。波段数和源文件波段数相同，其余可选参数建议查询后按需填写。

打开环境配置，此处可以配置 `输出坐标系、处理范围` 等，如果需要修改合并之后的坐标系，或者自定义输出文件的地理范围，都可以在此处配置。

> 该方法可以用于处理少量数据的合并，设计到大量数据请使用下一个方法。

## 二、使用 gdal 库进行合并

```bash
gdalwarp input_1.tif input_2.tif -dstnodata 0 -t_srs EPSG:4326 -te -180 -90 180 90 -tr 0.25 0.25 -r near -multi -wo "NUM_THREADS=ALL_CPUS" -co COMPRESS=LZW -co BIGTIFF=YES global_output.tif
```

使用上述指令进行合并，`-dstnodata` 指定 nodata 值，`-t_srs` 指定投影，`-te` 指定处理范围，`-tr` 指定分辨率，`-r` 指定插值方法，其余保持不变，最后指定输出文件名。

> 此处不推荐使用 `gdal_merge.py`，因为其占用内存较高。

可以修改 `-r` 参数指定不同的采样方法，如果需要还可以加入 -`srcnodata 0` 参数来指定重采样时不要处理的原始数据中的无用值。
