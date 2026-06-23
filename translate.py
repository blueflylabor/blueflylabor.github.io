import os
import frontmatter
import ollama

# ===================== 配置区 =====================
SOURCE_DIR = "_posts"
OUT_ROOT = "en/_posts"
MODEL_NAME = "translategemma:4b"
FROM_LANG = "中文"
TO_LANG = "English"
# ==================================================

os.makedirs(OUT_ROOT, exist_ok=True)

def llm_translate(text_chunk):
    prompt = f"""
你是严格的Markdown翻译工具，只输出译文，不要任何多余文字、解释、开场白。
规则：
1. 将所有中文自然语言准确翻译成流畅英文；
2. 绝对禁止修改、删除、翻译以下内容：
   - ``` 代码块内全部内容
   - $...$ 单行数学公式、$$...$$ 多行数学公式，公式里所有LaTeX符号、反斜杠、数字原样不动
   - Markdown标记：# * - > []() | ` 等所有格式符号
3. 保留原文所有换行、空行、段落结构，排版不能变；
4. 只翻译人类可读的中文句子，代码、公式、符号100%原样复制。

原文段落：
{text_chunk}
"""
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp["message"]["content"].strip()

def process_markdown(md_text):
    lines = md_text.splitlines()
    output = []
    in_code = False
    in_math_block = False
    text_buf = []

    for line in lines:
        strip_line = line.strip()
        # 代码块边界
        if strip_line.startswith("```"):
            if text_buf:
                output.extend(llm_translate("\n".join(text_buf)).splitlines())
                text_buf.clear()
            output.append(line)
            in_code = not in_code
            continue
        if in_code:
            output.append(line)
            continue

        # 多行数学公式块 $$
        if strip_line.startswith("$$"):
            if text_buf:
                output.extend(llm_translate("\n".join(text_buf)).splitlines())
                text_buf.clear()
            output.append(line)
            in_math_block = not in_math_block
            continue
        if in_math_block:
            output.append(line)
            continue

        # 普通文本行（含行内$公式，交给模型识别保护）
        text_buf.append(line)

    # 处理剩余文本
    if text_buf:
        output.extend(llm_translate("\n".join(text_buf)).splitlines())

    return "\n".join(output)

# 遍历所有中文md
for fname in os.listdir(SOURCE_DIR):
    if not fname.endswith(".md") or fname.endswith(".en.md"):
        continue
    src_path = os.path.join(SOURCE_DIR, fname)

    with open(src_path, "r", encoding="utf-8") as f:
        meta, raw_content = frontmatter.parse(f.read())

    # 翻译标题中文
    if "title" in meta:
        meta["title"] = llm_translate(meta["title"])

    # 翻译正文，隔离代码/大块公式
    en_content = process_markdown(raw_content)

    # 移除tags
    if "tags" in meta:
        del meta["tags"]

    # 多语言标识
    meta["lang"] = "en"
    meta["ref"] = fname.replace(".md", "")

    # 写出英文文件
    en_post = frontmatter.Post(en_content, **meta)
    out_path = os.path.join(OUT_ROOT, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(en_post))

    print(f"✅ 生成英文文件: {out_path}")