---
title: '修改 ssh 端口解决连接 Github 超时问题'
date: 2024-09-22T11:00:16+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - fixup
---





某一天突然不能使用 SSH 连接到 Github 了，无论是 clone 还是 push 都不行，报错如下：

```ba
ssh: connect to host github.com port 22: Connection timed out
```

也不知道是网络问题还是开启了 2FA，亦或者是防火墙的原因？但是一番排查都没有解决问题，随后在 StackFlow 上发现了如下配置方法。

在 ~/.ssh 目录下创建 config 文件，内容如下：

```
Host github.com
 Hostname ssh.github.com
 Port 443
```

问题解决了，看起来是 22 端口被防火墙屏蔽了，但是我根本没开防火墙。。。
