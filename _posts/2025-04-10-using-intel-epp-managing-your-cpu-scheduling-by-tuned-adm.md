---
layout: post
title: "intel-epp-packaging: managing your intel-cpu scheduling by [tuned-adm]"
categories:
  - linux
  - intel
  - OS
---

# intel-tuned-epp-packaging-tools
[intel epp packaing](https://www.intel.com/content/www/us/en/developer/articles/tool/epp-tuning-package-linux.html)

# install
### for non-debian linux[Archlinux]
- [download tar.xz and unzip](https://www.intel.com/content/www/us/en/download/819707/epp-tuning-package-for-linux-on-intel-core-ultra-mobile-processor.html)
- Execute deploy.sh script from terminal or copy tuneD[tuned-adm] custom profiles manually to /etc/tuned/ folder
### for debian linux[Debian, Ubuntu]
- first time
```bash
wget -O- https://apt.repos.intel.com/clientbkm/pub1/debian/public.gpg \ | gpg --dearmor | sudo tee /usr/share/keyrings/clientbkm-pub-archive-keyring.gpg > /dev/null

echo "deb [arch=any signed-by=/usr/share/keyrings/clientbkm-pub-archive-keyring.gpg] https://apt.repos.intel.com/clientbkm/pub1/debian stable main" | sudo tee /etc/apt/sources.list.d/clientbkm-pub-debian.list 

sudo apt update​
```
- install package
```bash
sudo apt install debkit-pkg.OPT.EPPprofile
```

# "tuned-adm profile <name for profile>" to activate profile 

- Dependency for tuneD: The system tuning daemon is needed to activate profiles. It is installed from the OS distro repo. 
- Note: Installing tuneD will disable Power Profile Daemon (PPD) on Ubuntu* systems, also Arch* systems.

# tuned-adm
You can use tuned-adm manage your intel-cpu[after intel 6th generation supported] scheduling
```bash
$ tuned-adm profile intel-best_performance_mode
$ tuned-adm profile intel-best_power_efficiency_mode
```
show activate profile
```bash
tuned-adm activate
```
```bash
$ tuned-adm -h                                                               
usage: tuned-adm [-h] [--version] [--debug] [--async] [--timeout TIMEOUT] [--loglevel LOGLEVEL]
                 {list,active,off,profile,profile_info,recommend,verify,auto_profile,profile_mode,instance_acquire_devices,get_instances,instance_get_devices} ...

Manage tuned daemon.

positional arguments:
  {list,active,off,profile,profile_info,recommend,verify,auto_profile,profile_mode,instance_acquire_devices,get_instances,instance_get_devices}
    list                list available profiles or plugins (by default profiles)
    active              show active profile
    off                 switch off all tunings
    profile             switch to a given profile, or list available profiles if no profile is given
    profile_info        show information/description of given profile or current profile if no profile is
                        specified
    recommend           recommend profile
    verify              verify profile
    auto_profile        enable automatic profile selection mode, switch to the recommended profile
    profile_mode        show current profile selection mode
    instance_acquire_devices
                        acquire devices from other instances and assign them to the given instance
    get_instances       list active instances of a given plugin or all active instances if no plugin is
                        specified
    instance_get_devices
                        list devices assigned to a given instance

options:
  -h, --help            show this help message and exit
  --version, -v         show program's version number and exit
  --debug, -d           show debug messages
  --async, -a           with dbus do not wait on commands completion and return immediately
  --timeout, -t TIMEOUT
                        with sync operation use specific timeout instead of the default 600 second(s)
  --loglevel, -l LOGLEVEL
                        level of log messages to capture (one of debug, info, warn, error, console, none).
                        Default: console

```

# customized setttings

- modify files in pkg.OPT.EPPprofile-1.05.240206.1-x86_64/profiles
- ``` sudo deploy.sh ```
- maybe failed to upload your customized profile, cp /profile/* /usr/lib/tuned/profiles
- tuned-adm profile <your customized profile>

### example settings
##### **intel-extreme_performance_mode**
- look at script.sh
```bash
#!/bin/sh

#Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: GPL-2.0-or-later

#EPP_PATH "/sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference"
#EPB_PATH "/sys/devices/system/cpu/cpu$i/power/energy_perf_bias"

. /usr/lib/tuned/functions

start() {
    n=$(nproc)
    i=0
    while [ "$i" -lt $n ]; do
        echo 64 | sudo tee  /sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference
        i=$(( i + 1 ))
    done 
    return 0
}

stop() {   
    n=$(nproc)
    i=0
    while [ "$i" -lt $n ]; do
        echo "balance_performance" | sudo tee  /sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference
        i=$(( i + 1 ))
    done     
    return 0
}

process $@

```

- edited script.sh
``` echo [customized num] | sudo tee  /sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference``` edit a customized num in 0~255 range  
    - 0<--  : release cpu performance
    - -->255: limit cpu performance
    - 0~64 and 255-64~255 relative okay
- after script.sh
```bash
#!/bin/sh

#Copyright (C) 2024 Intel Corporation
#SPDX-License-Identifier: GPL-2.0-or-later

#EPP_PATH "/sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference"
#EPB_PATH "/sys/devices/system/cpu/cpu$i/power/energy_perf_bias"

. /usr/lib/tuned/functions

start() {
    n=$(nproc)
    i=0
    while [ "$i" -lt $n ]; do
        echo 0 | sudo tee  /sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference
        i=$(( i + 1 ))
    done 
    return 0
}

stop() {   
    n=$(nproc)
    i=0
    while [ "$i" -lt $n ]; do
        echo "balance_performance" | sudo tee  /sys/devices/system/cpu/cpu$i/cpufreq/energy_performance_preference
        i=$(( i + 1 ))
    done     
    return 0
}

process $@

```
- edited profile subname [main] in tuned.conf[wiil see in display of **```tuned-adm list```**]
```
[main]
summary=Intel epp 0 TuneD profile

[script]
script=${i:PROFILE_DIR}/script.sh
```
- after custmizing, ```sudo deploy.sh```, cp profiles to ```/usr/lib/tuned/profiles```

# Writing a mangement gui included hidden tray
Using qt6 and python3 
```python
import subprocess
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox


def get_power_status():
    try:
        output = subprocess.check_output(['upower', '-i', subprocess.check_output(
            ['upower', '-e'], text=True).strip().split('\n')[0]]).decode('utf-8')
        for line in output.split('\n'):
            if 'state:' in line:
                return line.split(':')[1].strip()
    except Exception as e:
        QMessageBox.critical(None, "错误", f"获取电源状态时出错: {e}")
    return None


def get_current_mode():
    try:
        output = subprocess.check_output(['tuned-adm', 'active']).decode('utf-8')
        return output.split()[-1]
    except Exception as e:
        QMessageBox.critical(None, "错误", f"获取当前模式时出错: {e}")
    return None


def switch_mode(mode):
    try:
        subprocess.run(['tuned-adm', 'profile', mode], check=True)
        QMessageBox.information(None, "成功", f"已成功切换到 {mode} 模式。")
        update_status()
    except subprocess.CalledProcessError as e:
        QMessageBox.critical(None, "错误", f"切换模式时出错: {e}")


def update_status():
    power_status = get_power_status()
    current_mode = get_current_mode()
    power_status_label.setText(f"当前充电状态: {power_status}")
    current_mode_label.setText(f"当前 tuned-adm 模式: {current_mode}")


app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("电源模式切换")

layout = QVBoxLayout()

# 创建标签显示当前状态
power_status_label = QLabel("当前充电状态: 正在获取...")
layout.addWidget(power_status_label)

current_mode_label = QLabel("当前 tuned-adm 模式: 正在获取...")
layout.addWidget(current_mode_label)

# 创建按钮用于切换模式
performance_button = QPushButton("切换到高性能模式")
performance_button.clicked.connect(lambda: switch_mode("intel-best_performance_mode"))
layout.addWidget(performance_button)

power_efficiency_button = QPushButton("切换到节能模式")
power_efficiency_button.clicked.connect(lambda: switch_mode("intel-best_power_efficiency_mode"))
layout.addWidget(power_efficiency_button)

window.setLayout(layout)

# 初始状态更新
update_status()

window.show()
sys.exit(app.exec())
    
```
### reference
https://www.intel.com/content/www/us/en/developer/articles/tool/epp-tuning-package-linux.html 
https://www.intel.com/content/www/us/en/download/819707/epp-tuning-package-for-linux-on-intel-core-ultra-mobile-processor.html