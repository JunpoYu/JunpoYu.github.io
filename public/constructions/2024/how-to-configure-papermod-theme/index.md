# 如何配置 PaperMod 主题




所有的配置信息都填写在网站根目录下的 `hugo.yaml` 文件中。

## 一、配置文件

```yaml
baseURL: https://jespery.github.io/ # 自己网站的 url
languageCode: zh-cn # 语言
title: RubySIU's Blog # 网站名
theme: ["PaperMod"] # 使用的主题

enableInlineShortcodes: true #允许内联短码
enableEmoji: true # 允许使用 Emoji 表情，建议 true
enableRobotsTXT: true # 允许爬虫抓取到搜索引擎，建议 true

hasCJKLanguage: true # 自动检测是否包含 中文日文韩文 如果文章中使用了很多中文引号的话可以开启

buildDrafts: false # 是否构建 drafts 文章
buildFuture: false # 是否构建 future 文章
buildExpired: false  # 是否构建 expired 文章

paginate: 15 # 每页文章数

minify:
    disableXML: true # 

# defaultContentLanguage: zh # 最顶部首先展示的语言页面
# defaultContentLanguageInSubdir: true

outputs: # 此部分用于搜索页面
  home:
    - HTML
    - RSS
    - JSON

menu: # 主页右上角菜单栏
  main:
    - identifier: search # 标识
      name: 🔍搜索 # 菜单名
      url: search/ # 对应 url
      weight: 1 # 权重，用于排序
    - identifier: archives
      name: ⏱时间轴
      url: archives/
      weight: 2

params: # 参数
  defaultTheme: auto # 自动切换亮暗主题
  disableThemeToggle: false # 启用亮暗切换按钮
  ShowAllPagesInArchive: true # 在 Archive 页面显示所有文件夹下的文章

  profileMode: # 使用 profile 模式
    enabled: true 
    title: "RubySIU's Blog" # profile 主页的大标题
    # subtitle: ""
    imageUrl: "chikaflare_icon.jpg" # 主页头像
    imageTitle: "奇可芙蕾雅！" # 头像文字
    imageWidth: 120 # custom size
    imageHeight: 120 # custom size
    buttons: # 主页按钮
      - name: 技术
        url: "/posts"
      - name: 搭建
        url: "/builds"

  socialIcons: # 主页社交入口
    - name: "email"
      url: "mailto:junpo.yu@foxmail.com"
    - name: "Github"
      url: "https://github.com/JesperY"
    - name: "RSS"
      url: "index.xml"

  ShowShareButtons: false # 不显示分享按钮，因为都是外国网站
  ShowReadingTime: true # 显示阅读时间 
  ShowBreadCrumbs: false # 不显示面包屑导航
  ShowPostNavLinks: true # 显示上一页、下一页
  ShowCodeCopyButtons: true # 显示代码复制按钮


  fuseOpts: # 搜索配置，照搬
      isCaseSensitive: false
      shouldSort: true
      location: 0
      distance: 1000
      threshold: 0.4
      minMatchCharLength: 0
      # limit: 10 # refer: https://www.fusejs.io/api/methods.html#search
      keys: ["title", "permalink", "summary", "content"]

  
```

## 二、修改时间轴页面为中文

此部分照搬 [Sulv's Blog](https://www.sulvblog.cn/posts/blog/hugo_archives_chinese/)，定位到 `themes/PaperMod/layouts/_default/archives.html`，建议将` layouts、i18n、assets`下的文件复制到站点根目录，以免更新主题导致覆盖。

修改 `archives.html` 为如下内容：

```html
{{- define "main" }}

<header class="page-header">
  <h1>{{ .Title }}</h1>
  {{- if .Description }}
  <div class="post-description">
    {{ .Description }}
  </div>
  {{- end }}
</header>

{{- $pages := where site.RegularPages "Type" "in" site.Params.mainSections }}

{{- if .Site.Params.ShowAllPagesInArchive }}
{{- $pages = site.RegularPages }}
{{- end }}

{{- range $pages.GroupByPublishDate "2006" }}
{{- if ne .Key "0001" }}
<div class="archive-year">
  <h2 class="archive-year-header">
    {{- replace .Key "0001" "" }}年<sup class="archive-count">&nbsp;&nbsp;{{ len .Pages }}</sup>
  </h2>
  {{- range .Pages.GroupByDate "January" }}
  <div class="archive-month">
    <h3 class="archive-month-header">
      {{- if eq .Key "December" }}
      {{ "12月" }}
      {{- end }}
      {{- if eq .Key "November" }}
      {{ "11月" }}
      {{- end }}
      {{- if eq .Key "October" }}
      {{ "10月" }}
      {{- end }}
      {{- if eq .Key "September" }}
      {{ "9月" }}
      {{- end }}
      {{- if eq .Key "August" }}
      {{ "8月" }}
      {{- end }}
      {{- if eq .Key "July" }}
      {{ "7月" }}
      {{- end }}
      {{- if eq .Key "June" }}
      {{ "6月" }}
      {{- end }}
      {{- if eq .Key "May" }}
      {{ "5月" }}
      {{- end }}
      {{- if eq .Key "April" }}
      {{ "4月" }}
      {{- end }}
      {{- if eq .Key "March" }}
      {{ "3月" }}
      {{- end }}
      {{- if eq .Key "February" }}
      {{ "2月" }}
      {{- end }}
      {{- if eq .Key "January" }}
      {{ "1月" }}
      {{- end }}
      <!-- {{- .Key }} -->
      <sup class="archive-count">&nbsp;&nbsp;{{ len .Pages }}
      </sup>
    </h3>
    <div class="archive-posts">
      {{- range .Pages }}
      {{- if eq .Kind "page" }}
      <div class="archive-entry">
        <h3 class="archive-entry-title">
          {{- .Title | markdownify }}
          {{- if .Draft }}<sup><span class="entry-isdraft">&nbsp;&nbsp;[draft]</span></sup>{{- end }}
        </h3>
        <div class="archive-meta">
          {{- partial "post_meta.html" . -}}
        </div>
        <a class="entry-link" aria-label="post link to {{ .Title | plainify }}" href="{{ .Permalink }}"></a>
      </div>
      {{- end }}
      {{- end }}
    </div>
  </div>
  {{- end }}
</div>
{{- end }}
{{- end }}

{{- end }}{{/* end main */}}
```


