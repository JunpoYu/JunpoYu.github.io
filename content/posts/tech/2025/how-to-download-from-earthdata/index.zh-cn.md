---
title: '如何优雅地从 EarthData 下载数据'
date: 2025-01-16T19:13:23+08:00
draft: False
ShowToc: true
TocOpen: true
tags:
  - research skills
---





EarthData 是一个由 NASA 维护的数据门户网站，这里可以很方便地检索并下载地理或者气象数据，例如 MODIS 数据。

但是如果通过网页检索有一个弊端，就是 EarthData 会把检索到的所有文件的下载链接缓存到页面上，然后再生成下载元文件，当检索的数据量过大时，检索时间很长不说，大量的链接文本也会导致页面崩溃，因而前功尽弃。

本文提供一个通过 python 接口检索下载链接的方法，该方法不是基于爬虫，而是由 NASA 官方提供接口，所以不需要担心被限流等问题。

```python
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import time
import concurrent.futures

# Earthdata 登录信息
USERNAME = "Your Earthdata Account"  # 替换为您的 Earthdata 用户名
PASSWORD = "Your Earthdata Password"  # 替换为您的 Earthdata 密码

# 检查年份格式
# 如果希望更加细致的自定义时间，可以参考下面 params 中的 temporal 字段构建方法
start_year = 2000 # 下载时间的起始年
end_year = 2001 # 下载时间的结束年

# API URL 和参数
base_url = "https://cmr.earthdata.nasa.gov/search/granules.json" # NASA 提供的 CMR 查询接口

# 获取下载链接的函数
def get_granules(start_year, end_year):
    granules = []
    params = {
        "short_name": "MOD16A2GF",  # 数据集名称
        "version": "061",  # 数据集版本
        "temporal": f"{start_year}-01-01T00:00:00Z,{end_year}-12-31T23:59:59Z",  # 时间范围
        "page_size": 2000,  # 每页返回记录数
    }
    page = 1 # 分页查询
    while True:
        print(f"Fetching page {page} for {year}...")
        response = requests.get(base_url, params={**params, "page_num": page})
        if response.status_code != 200:
            print(f"Failed to fetch granules for {year}, page {page}: {response.status_code}")
            break
        data = response.json()
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            print(f"No entries found for {year} on page {page}. Exiting.")
            break
        for entry in entries:
            for link in entry["links"]:
                if "href" in link and link["href"].endswith(".hdf") and link['href'].startswith("http"):
                    granules.append(link["href"])  # 提取 .hdf 文件链接
        page += 1
    return granules # 返回链接列表

get_granules(satrt_year, end_year)
```

该方法会返回一个列表，每个元素代表一个文件的下载链接。这些链接可以直接复制到浏览器下载，可以使用 Earthdata 提供的下载脚本，或者自己写一个 curl 或 wget 脚本下载。这里提供一个基于 python 的多进程下载脚本。

```python
import os
import sys
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import time
import concurrent.futures

# Earthdata 登录信息
USERNAME = "Your Earthdata Account"  # 替换为您的 Earthdata 用户名
PASSWORD = "Your Earthdata Password"  # 替换为您的 Earthdata 密码

# 检查年份格式
# 如果希望更加细致的自定义时间，可以参考下面 params 中的 temporal 字段构建方法
start_year = 2000 # 下载时间的起始年
end_year = 2001 # 下载时间的结束年

# API URL 和参数
base_url = "https://cmr.earthdata.nasa.gov/search/granules.json" # NASA 提供的 CMR 查询接口

# 获取下载链接的函数
def get_granules(year):
    """
    此处修改为分年下载，从而实现每年并行
    """
    granules = []
    params = {
        "short_name": "MOD16A2GF",  # 数据集名称
        "version": "061",  # 数据集版本
        "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",  # 时间范围
        "page_size": 2000,  # 每页返回记录数
    }
    page = 1 # 分页查询
    while True:
        print(f"Fetching page {page} for {year}...")
        response = requests.get(base_url, params={**params, "page_num": page})
        if response.status_code != 200:
            print(f"Failed to fetch granules for {year}, page {page}: {response.status_code}")
            break
        data = response.json()
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            print(f"No entries found for {year} on page {page}. Exiting.")
            break
        for entry in entries:
            for link in entry["links"]:
                if "href" in link and link["href"].endswith(".hdf") and link['href'].startswith("http"):
                    granules.append(link["href"])  # 提取 .hdf 文件链接
        page += 1
    return granules # 返回链接列表

# 下载文件的函数
def download_file(url, download_dir, retries=3):
    try:
        local_filename = os.path.join(download_dir, url.split("/")[-1])  # 从 URL 提取文件名
        if os.path.exists(local_filename):
            print(f"File {local_filename} already exists. Skipping download.")
            return
        print(f"Downloading {url} to {local_filename}...")
        with requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD), stream=True) as response:
            response.raise_for_status()  # 如果请求失败，将引发异常
            with open(local_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Download complete: {local_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if retries > 0:
            print(f"Retrying download for {url}...")
            time.sleep(2)  # 延迟 2 秒后重试
            download_file(url, download_dir, retries - 1)
        else:
            print(f"Failed to download {url} after retries.")

# 主函数，用于执行并行下载
def download_granules_for_year(year, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)  # 创建下载目录
    granules = get_granules(year)
    print(f"Found {len(granules)} granules for {year}.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 使用线程池并行下载
        futures = [executor.submit(download_file, url, download_dir) for url in granules]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 阻塞，直到任务完成

if __name__ == "__main__":
    # 允许用户指定下载目录
    download_dir = "Your Data Storage Dir"
    
    # 对于每个年份并行下载
    for year in range(start_year, end_year + 1):
        print(f"Starting downloads for {year}...")
        download_granules_for_year(year, os.path.join(download_dir, str(year)))
```

