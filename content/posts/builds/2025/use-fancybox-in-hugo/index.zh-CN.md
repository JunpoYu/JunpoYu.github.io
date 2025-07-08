---
title: '在 Hugo 中使用 Fancybox 实现图片点击放大'
date: 2025-05-23T20:09:41+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - hugo
---







由于原生 Hugo 的页面不支持点击图片放大，而有时候我在博客中会使用一些内容比较多的图片，这就导致图片上的内容很小看不清。

有一种方法可以通过修改 Mark Render Hooks 实现 fancybox 方法图片，具体实现方法来自 [nan0in's blog](https://www.nan0in27.cn/p/hugo%E5%9B%BE%E7%89%87%E7%82%B9%E5%87%BB%E6%94%BE%E5%A4%A7/)，在此转载以备日后查阅。

### 1. 修改 hugo.yaml

总之就是 Hugo 的配置文件，也可能你的站点采用其他格式。

```yaml
params:
	fancybox: true
```

添加一个 `fancybox: true` 字段即可。

### 2. 修改 render-image.html

该文件通常在站点目录下的 `/layouts/\_default/\_markup/render-image.html` ，修改文件内容为以下内容：

```html
{{if .Page.Site.Params.fancybox }}
<div class="post-img-view">
<a data-fancybox="gallery" href="{{ .Destination | safeURL }}">
<img src="{{ .Destination | safeURL }}" alt="{{ .Text }}" {{ with .Title}} title="{{ . }}"{{ end }} />
</a>
</div>
{{ end }}

```

### 3. 在 footer.html 或者 header.html 中导入

```html
{{if .Page.Site.Params.fancybox }}
<script src="https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/fancyapps/fancybox@3.5.7/dist/jquery.fancybox.min.css" />
<script src="https://cdn.jsdelivr.net/gh/fancyapps/fancybox@3.5.7/dist/jquery.fancybox.min.js"></script>
{{ end }}

```

以上。

> 注意修改的都是 Hugo 站点的文件，主题文件夹中也有类似的内容，但是修改主题的文件是不起作用的。
