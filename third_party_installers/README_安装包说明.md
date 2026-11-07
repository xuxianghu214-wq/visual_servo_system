# 第三方离线安装包

- `Python_3.11.9_x64`：Python 官方 Windows x64 嵌入式运行时（首选）和安装程序（兼容后备）。
- `Teledyne_Spinnaker_4.4.0.246_x64`：从本机已安装版本的 Windows Package Cache 复制的完整安装入口和四个 MSI 组件。
- 每个目录中的 `SHA256SUMS.txt` 用于复制后完整性校验。

普通 Python 环境安装不需要管理员权限。相机驱动安装会修改系统驱动与 GenTL 环境，必须由用户主动运行根目录的 `安装相机驱动.ps1` 并确认管理员授权。
