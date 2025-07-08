---
title: 'hugo 博客 latex 公式兼容'
date: 2024-11-01T20:57:03+08:00
draft: false
ShowToc: true
TocOpen: true
tags:
  - hugo
categories:
  - hugo
---

技术博客逃不了写公式，$LaTEX$必然是最好的选择，但是 hugo 对 LaTEX 的兼容并不好，因此采用 MathJax 渲染。

最简单的做法是将以下代码插入到 `layouts\partials\footer.html`，也可以用自己喜欢的方式将 js 代码添加到页面 html 中。

```javascript
<script type="text/javascript"
        async
        src="https://cdn.bootcss.com/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    displayMath: [['$$','$$'], ['\[\[','\]\]']],
    processEscapes: true,
    processEnvironments: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    TeX: { equationNumbers: { autoNumber: "AMS" },
         extensions: ["AMSmath.js", "AMSsymbols.js"] }
  }
});

MathJax.Hub.Queue(function() {
    // Fix <code> tags after MathJax finishes running. This is a
    // hack to overcome a shortcoming of Markdown. Discussion at
    // https://github.com/mojombo/jekyll/issues/199
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>

<style>
code.has-jax {
    font: inherit;
    font-size: 100%;
    background: inherit;
    border: inherit;
    color: #515151;
}
</style>
```

