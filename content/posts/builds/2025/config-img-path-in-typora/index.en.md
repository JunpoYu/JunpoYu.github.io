---
title: 'Config Img Path in Typora'
date: 2025-05-04T11:14:40+08:00
draft: true
ShowToc: true
TocOpen: true
tags:
  - writing
  - typora
---

## Configure the relative path of images in Typora

To embed images in your self-built blog, there are two typically methods: using an **image hosting service** or a **local relative path**. There is no doubt that the image hosting service is the simplest and most convienient way. Many free services are also avaiable. Howere, some people are always concerned about the security of their data when entrusting it to others. Howere, when inserting images using a Markdown editor (like Typora), you often to consider the incompatibility between the blog's build path and the editor'e default image referencing. Fourtunately, Typora has started supporting the use of font matter to customize the images preview path within Markdown files,  without affecting the blogs genration process.

> This article only discuss posts written in markdown syntax. Hogo shortcodes like `{{method...}}` (or any other shortcodes not directly supported by Markdown editors) are not within the scope of this discussion, as they cannot be previewed dierctly in Markdown editors.

### 一、博客构建的路径

不同的博客系统构建方式也不同，本博客使用 Hugo 构建，因此以 Hugo 为例。

Hugo 将博文 md 文件存放在 content 目录下，该目录下可以根据需要自行创建多个子目录，从而对 md 文件进行分类。生成文章页面的 md 文件也有两种组织方式，一种是直接以文章标题作为 md 文件名，每个文章一个 md 文件，同一分类下的 md 文件处于同一层级。文章所需要引用的资源文件存放在和 content 同级的 static 目录下。

```
content
	|-- subdir
		|-- blog1.md
		|-- blog2.md
		...
...
static
	|--my_resources
		|-- img.jpg
		|-- video.mp4
```

另一种是采用页面包构建文章，也就是将文章名作为目录名，目录下使用 `index.md` 撰写文章内容。这种方法的好处是可以将资源文件直接存放在页面包中。

```
content
	|-- subdir
		|-- blog
			|-- index.md
			|-- img.jpg
		|-- blog2
			|-- index.md
			|-- video.mp4
		...
...
static
	|--...
```

接下来分别描述这两种不同构建方式下如何使用资源文件



## 二、基于 文章名.md 的构建

执行构建指令 `hugo build` 之后，首先会将 content 目录下的内容编译为静态页面，即文章名作为目录名，md 文件转为 index.html。文章中引用资源的路径也原封不同，因此通常推荐使用相对路径的方式在文章中引用资源文件。而编译后的静态页面会以相同的目录结构全部存放在 public 路径下，即 `content/subdir/blog1.md` 会变成 `public/subdir/blog2.md`。存放在 static 目录下的资源文件则会以相同的目录结构复制到 public 目录下。

```
content
	|-- subdir
		|-- blog1.md
		|-- blog2.md
		...
...
static
	|--my_resources
		|-- img.jpg
		|-- video.mp4

public
	|-- subdir
		|-- blog1
			|-- index.html
		|-- blog2
			|-- index.html
	|-- my_resources
		|-- img.jpg
		|-- vidio.mp4
```

因此像这种目录结构，我们引用资源时可以使用相对路径，注意此处的相对路径是指以 public 为根目录的相对路径。例如 blog1.md 中引用 img.jpg 时，引用路径应该是 `../../my_resources/img.jpg`。但是我们在撰写 md 文件时是在 content 目录下，因此这种相对路径的引用方式依然无法实现在编辑器中直接预览。

这里我们需要使用绝对路径并且借用 Typora 中的一项配置 `typora-root-url` 来实现预览效果，同时不影响文章的构建。

博客构建并发布后，在文章中引用资源的根目录 `/` 其实就是 `public` 路径，因此发布之后的 img.jpg 文件的绝对路径就是 `/mysources/img.jpg`，我们可以直接在文章中使用这个绝对路径来引用资源而不会出错，同时我们在文章的 `font-matter` 中添加一项配置。

```
---
...other font-matter
typora-root-url: ..\..\tatic\
---
```

`typora-root-url` 决定了 Typora 如何对待 md 文件中出现的绝对路径的根目录。即 Typora 在渲染引用的资源文件时，会将绝对路径的根目录 `/` 替换为 `typora-root-url`，此时我们引用的 `/mysources/img.jpg` 就变成了 `../../static/mysources/img.jpg`。此时 Typora 会将 img.jpg 作为一个相对路径资源从而正确渲染，而在编译时 `typora-root-url` 不会生效，hugo 会将 `/mysources/img.jpg` 作为以 public 为根目录的绝对路径进行构建，因此构建过程也不会出错。

> font-matter 中使用反斜杠是因为这是在 windows 上，文章中的目录都使用 linux 风格，实际上都是一个意思。

## 三、基于页面包 index.md 的构建

基于页面包的构建更加方便，根本不需要涉及上面的转换过程。在生成新的文章时使用 `hugo new posts/new-post/index.md` 指令，然后将所有和文章相关的资源放在同步目录下，直接引用即可。构建过程和 Typora 的预览的路径都是正确的。
