---
title:  "ruby mirrors for Jekyll"
date:   2024-07-18
last_modified_at: 2024-07-18
categories: [Jekyll, 环境配置]
---

## look for source list
```
gem sources list
gem sources -l
```

## default mirror
```
https://rubygems.org/
```

### delete default and add cn mirrors
```
gem sources --remove https://rubygems.org/
gem sources --add https://gems.ruby-china.com/
```

~~gem sources --add https://ruby.taobao.org~~