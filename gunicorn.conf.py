import multiprocessing

# Render 自动注入 PORT，但设置默认值
bind = "0.0.0.0:10000"

# 免费实例内存小，1个 worker 足够，用多线程处理并发
workers = 1
threads = 4

# 关键：音频分析耗时，必须设为 120秒，否则 Render 会杀掉进程
timeout = 120

loglevel = 'info'