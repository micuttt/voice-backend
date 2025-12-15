# gunicorn.conf.py
workers = 2 # Render 免费实例 CPU 有限，2个 worker 比较合适
threads = 4
timeout = 120 # 音频分析比较耗时，增加超时时间
bind = "0.0.0.0:10000"