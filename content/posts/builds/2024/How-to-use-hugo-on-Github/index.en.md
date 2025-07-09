---
title: 'Use Hugo and GitHub to deploy website'
date: 2024-09-12T20:51:24+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - hugo
categories:
  - hugo
---

## 1. Installing Hugo on Windows

### 1. Prepare

Firstly you will need to install Git and Go. There are plenty of tutorials availables for this process.

### 2. Installation using pre-built binary package

Download the latest binary from the [Hugo release page](https://github.com/gohugoio/hugo/releases/latest), extract it to an appropriate directory, and add that directory to your `PATH` environment variable.

## 2. Build the website and set a theme

> It is recommended to use Git Bash.

### 1. Build the website

Use the following command to build the website.

```bash
hugo new site your_blog_path
```

### 2. Set a theme

**This is just one way to do it, it is recommended to follow the installation document provided by the theme you choose.**

The method recommended by the [PaperMod](https://adityatelange.github.io/hugo-PaperMod/) theme is as follows:

```bash
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
git submodule update --init --recursive # needed when you reclone your repo (submodules may not get cloned automatically)
```

> Alternatively, you can install the theme using `git clone`, a zip file or Hugo module.

Edit the configuration file `hugo.yaml` and add `theme: ["PaperMod"]`

## 3. Create your first post

### 1. create post

Create a post using command `hugo new posts/first-post.md` . The aritcle will be created in the `contents` directory.

If you want to add an image to your post, the root directory for image is the `static` directory. For example, If you reference an image using path `/posts/first-post/img.jpg`  , the actual file should be placed at `static/posts/first-post/img.jpg`.

### 2. Preview locally

Run command `hugo server -D`  to start a local server, and visit `localhost:1313` in your browser to preview your website. The `-D` flag tells Hugo to include content marked as draft.

> The `draft` field in font matter indicates whether a post is a draft, and the draft posts are exclude from build by default.

## 4. Use Github Actions to deploy your website to GitHub Pages

Why not just push `public` directory to your `github.io` repository? Because I do not want to maintain a separate repository to synchronize the entire Hugo site.

By using Github Actions, you can upload all your site files to your `github.io` repository, and then automatically build your Pages with Actions.

### 1. Add workflow

Create a new file  `.github/workflows/hugo.yaml` with the following content:

> 下面的文件为自用文件，请修改 branches 和 HUGO_VERSION，如果使用 submodule 方式添加 themes，则 deploy.steps 下的 uses 和 with 字段请保留。
>
> The file below is for my personal use. If you want to use it, please modify `branched` and `HUGO_VERSION`  fields. Also if you add theme using `submodule` method, please make sure to keep the `uses` and `with` fields under `deploy.steps`.

```yaml
# Sample workflow for building and deploying a Hugo site to GitHub Pages
name: Deploy Hugo site to Pages
​
on:
  # Runs on pushes targeting the default branch
  push:
    branches:
      - main
​
  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:
​
# Sets permissions of the GITHUB_TOKEN to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write
​
# Allow only one concurrent deployment, skipping runs queued between the run in-progress and latest queued.
# However, do NOT cancel in-progress runs as we want to allow these production deployments to complete.
concurrency:
  group: "pages"
  cancel-in-progress: false
​
# Default to bash
defaults:
  run:
    shell: bash
​
jobs:
  # Build job
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.134.2
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb          
      - name: Install Dart Sass
        run: sudo snap install dart-sass
      - name: Checkout
        uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0
      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v5
      - name: Install Node.js dependencies
        run: "[[ -f package-lock.json || -f npm-shrinkwrap.json ]] && npm ci || true"
      - name: Build with Hugo
        env:
          HUGO_CACHEDIR: ${{ runner.temp }}/hugo_cache
          HUGO_ENVIRONMENT: production
          TZ: America/Los_Angeles
        run: |
          hugo \
            --gc \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/"          
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public
​
  # Deployment job
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v2
        with:
          submodules: true
          fetch-depth: 0
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 2. Enable Action

Go to the `setting/pages` section in your `github.io` repository. Under `Build and deployment`, set `Sourec` to `GitHub Actions`.

![修改 Pages 为 Actions 生成](1.png)

### 3. Commit changes.

Upload the all of your Hugo website files t  theo `github.io` repository and `GitHub Action` will automatically run to build your site. You can view the build status on the repository's `Actions` page.
