---
layout: post
title: "intel-epp-packaging: managing your intel-cpu scheduling by [tuned-adm]"
categories: linux
---

qt6 and py3
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
