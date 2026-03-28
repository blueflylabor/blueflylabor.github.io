---
layout: post
title: "without updating the macOS to figure out the Markdown import to Mac Note app"
categories: solution
tags:
  - MacOS
  - OSbug
  - tools
---

## **without updating the macOS to figure out the Markdown import to Mac Note app**

If you’ve ever tried to get a clean Markdown file—with embedded images—into Apple’s Notes app (备忘录), you probably discovered that it’s a surprisingly broken experience. Notes supports rich text, sure, but Markdown? Not really. And unless you’re on the newest macOS (which you might not want to update for various perfectly valid reasons), there’s no direct way to import Markdown content, especially when it includes images.

Here’s the route that actually works—no updates, no iCloud gymnastics, and no copy-paste formatting disasters.

---

### 🧭 Step 1: Export Discord chats as Markdown

If your source is Discord, you can use [**DiscordChatExporter**](https://github.com/Tyrrrz/DiscordChatExporter). It’s a great little tool that lets you download your chat logs in several formats, including Markdown.

Command-line version example:

```bash
DiscordChatExporter.Cli export -t YOUR_TOKEN -c CHANNEL_ID -f Markdown -o chat.md
```

You’ll get a clean `.md` file, but all images will still be remote links like:

```markdown
![image](https://cdn.discordapp.com/attachments/...)
```

You can leave them as-is if you just want the text and image references, or download them locally if you want embedded images in the final document.

---

### ⚙️ Step 2: Convert Markdown to DOCX with Pandoc

Pages doesn’t natively import Markdown with images, but it handles `.docx` beautifully. Pandoc does the heavy lifting.

In your Markdown directory, run:

```bash
pandoc chat.md --resource-path=. -o chat.docx
```

This command:

* Reads your `chat.md`
* Pulls in any local image files
* Embeds them directly into the `.docx`
* Keeps text formatting consistent with your Markdown

If your images are stored in a subfolder (say `images/`), use:

```bash
pandoc chat.md --resource-path=images -o chat.docx
```

---

### 📄 Step 3: Open the DOCX in Pages

Now open `chat.docx` with **Pages** (either drag it in, right-click → Open with Pages, or double-click if that’s your default).

Pages will automatically convert it into its own format, and—voilà—your Markdown is rendered, styled, and the images are embedded properly.

---

### 🪄 Step 4: Copy everything into Notes (备忘录)

From Pages, simply **Select All → Copy → Paste** into Notes.

Because Pages uses Apple’s rich text format internally, everything pastes into Notes **with correct formatting**:

* Bold, italics, headings
* Lists
* Embedded images

No broken Markdown symbols, no invisible images, no plain text hell.

---

### 🧠 Why this works

Apple Notes doesn’t parse Markdown—it just accepts **RTF**-style rich text from Pages.
Pandoc’s DOCX export embeds images in a way Pages can read natively, making it the perfect “bridge format.”
This bypasses the need to rely on any macOS update or third-party converters that only half-work.

---

### 🧰 Optional enhancements

* **Keep images offline**: If you want local images rather than Discord-hosted ones, you can use a simple script or browser extension to download attachments before conversion.
* **Batch export**: For multiple channels or Markdown files, you can loop through them in your shell script:

  ```bash
  for f in *.md; do
      pandoc "$f" --resource-path=. -o "${f%.md}.docx"
  done
  ```
* **Automate Notes import**: If you want full automation, you could even use AppleScript to paste into Notes—but that’s another adventure.

---

### 🧩 Final Thoughts

This pipeline—
`Discord → Markdown → DOCX → Pages → Notes`
—might look circuitous, but it’s fast, stable, and most importantly: it **doesn’t require updating macOS** just to fix a half-broken import feature.

You can think of it as using Pages as a Markdown interpreter that Apple never shipped. Sometimes the most elegant solutions are the improvised ones.
