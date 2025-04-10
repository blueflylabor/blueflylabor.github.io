---
layout: page
title: 代码展示
permalink: /code-display/
---

<h1>加载的代码</h1>
<pre><code id="code-container" class="language-python"></code></pre>

<script>
  // 替换为你实际的 Raw 链接
  const rawUrl = 'https://raw.githubusercontent.com/blueflylabor/deviceTree/refs/heads/main/device.md';
  const codeContainer = document.getElementById('code-container');

  fetch(rawUrl)
    .then(response => response.text())
    .then(data => {
      codeContainer.textContent = data;
      Prism.highlightElement(codeContainer);
    })
    .catch(error => {
      console.error('加载代码时出错:', error);
    });
</script>