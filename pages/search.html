---
layout: default
title: Search
permalink: /search/
---
<style>
.search-input{width:100%;padding:10px;font-size:16px;border:1px solid #ddd;border-radius:6px;box-sizing:border-box}
.search-item{padding:12px 0;border-bottom:1px solid #eee}
.search-item h3{margin:0 0 4px 0}
.search-item .meta{color:#777;font-size:14px}
.empty-tip{color:#999;text-align:center;padding:30px 0}
</style>

<div class="page-container">
  <h1>Search Articles</h1>
  <input class="search-input" id="searchInput" placeholder="Input keyword to search title, tags, content..." autocomplete="off">
  <div id="resultBox"></div>
</div>

<script>
// 全局文章库
let articleList = [];
const input = document.getElementById("searchInput");
const resultBox = document.getElementById("resultBox");

// 加载索引
async function loadIndex(){
  const res = await fetch("{{ site.baseurl }}/search-data.json");
  articleList = await res.json();
}

// 关键词匹配逻辑
function search(keyword){
  const kw = keyword.toLowerCase().trim();
  if(!kw){
    resultBox.innerHTML = "";
    return;
  }
  const filter = articleList.filter(item=>{
    const t = item.title.toLowerCase();
    const c = item.content.toLowerCase();
    const tagStr = item.tags.join(" ").toLowerCase();
    const catStr = item.categories.join(" ").toLowerCase();
    return t.includes(kw) || c.includes(kw) || tagStr.includes(kw) || catStr.includes(kw);
  });
  render(filter);
}

// 渲染结果
function render(list){
  if(list.length === 0){
    resultBox.innerHTML = `<p class="empty-tip">No matching articles found</p>`;
    return;
  }
  let html = "";
  list.forEach(item=>{
    html += `
    <div class="search-item">
      <h3><a href="{{ site.baseurl }}${item.url}">${item.title}</a></h3>
      <div class="meta">Tags: ${item.tags.join(", ")} | Categories: ${item.categories.join(", ")}</div>
    </div>
    `;
  })
  resultBox.innerHTML = html;
}

// 监听输入
input.addEventListener("input",e=>search(e.target.value));
// 预加载索引
loadIndex();
</script>