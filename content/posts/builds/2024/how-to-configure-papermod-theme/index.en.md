---
title: 'How to configure PaperMod theme'
date: 2024-09-14T21:14:25+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - hugo
categories:
  - hugo
---



All configuration details are specified in the `hugo.yaml` file located in the website's root directory.

## 1. Configuration

```yaml
baseURL: https://jespery.github.io/ # your site's url
languageCode: zh-cn # language
title: RubySIU's Blog # your site's title
theme: ["PaperMod"] # enabled theme

enableInlineShortcodes: true # allow inline shortcodes
enableEmoji: true # allow Emoji
enableRobotsTXT: true # allow spider 

hasCJKLanguage: true # automatically detect the presence of Chinese, Japanese or Korean text

buildDrafts: false # whether to generate draft articles
buildFuture: false # whether to generate future articles
buildExpired: false  # whether to generate expired articles

paginate: 15 # number of articles per page

minify:
    disableXML: true # 

# defaultContentLanguage: zh # default language
# defaultContentLanguageInSubdir: true

outputs: # for search page
  home:
    - HTML
    - RSS
    - JSON

menu: # menu bar on the top right 
  main:
    - identifier: search #
      name: 🔍search # menu name
      url: search/ # corresponding url
      weight: 1 # weights for sort
    - identifier: archives
      name: ⏱time line
      url: archives/
      weight: 2

params:
  defaultTheme: auto # automatically switches between light and dark theme
  disableThemeToggle: false # enable the light/dark mode toggle button
  ShowAllPagesInArchive: true # display all articles from all directories on the arvhive paeg

  profileMode: # use profile mode
    enabled: true 
    title: "RubySIU's Blog" # title of profile page
    # subtitle: ""
    imageUrl: "chikaflare_icon.jpg" # icon
    imageTitle: "奇可芙蕾雅！" 
    imageWidth: 120 # custom size
    imageHeight: 120 # custom size
    buttons: # home page button
      - name: tech
        url: "/posts"
      - name: 搭建
        url: "/builds"

  socialIcons: # social media links on the home page
    - name: "email"
      url: "mailto:junpo.yu@foxmail.com"
    - name: "Github"
      url: "https://github.com/JesperY"
    - name: "RSS"
      url: "index.xml"

  ShowShareButtons: false #
  ShowReadingTime: true 
  ShowBreadCrumbs: false 
  ShowPostNavLinks: true 
  ShowCodeCopyButtons: true 


  fuseOpts: # configuration of search
      isCaseSensitive: false
      shouldSort: true
      location: 0
      distance: 1000
      threshold: 0.4
      minMatchCharLength: 0
      # limit: 10 # refer: https://www.fusejs.io/api/methods.html#search
      keys: ["title", "permalink", "summary", "content"]

  
```

## 2. Modify the timeline page to display in Chinese

This section is directly adapted from  [Sulv's Blog](https://www.sulvblog.cn/posts/blog/hugo_archives_chinese/). you need to find the `themes/PaperMod/layouts/_default/archives.html` file and copy the files under `layouts、i18n、assets`  directories to the root directory to prevent them from bing overwritten during theme updates.

Modify  `archives.html`  as follows:

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

