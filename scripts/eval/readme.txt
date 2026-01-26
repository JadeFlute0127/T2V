1. 把数据放在 input/dataset/video目录下
2. 执行 replace_video_name.py
    2.1 在 TARGET_MODEL_NAME 列表中添加文件名作为模型名（例如Sora2）
    2.2 一般有两种视频文件组织结构：
        （1）Sora2/Engineering、Sora2/Natural-Science、Sora2/Huaman、Sora2/Healthcare -> 这种一般是1500数据
        （2）Sora2/xxx.mp4 -> 这种一般是300数据
    2.3 执行replace_video_name.py
        正常来说会替换几个视频的名称，兼容上述两种视频文件组织结构
3. 执行video_file_check.py
    3.1 会以sample.jsonl为标准，检查是否缺少需要的视频
    3.2 如果缺少，则根据日志，补充对应id的视频到Sora2/patch目录下
4. 执行evaluation.py
    4.1 eval结果会在output/目录下
    4.2 跑完后，可以检查下output/Sora2-evaluation.jsonl是否是300行
    4.3 如果不是300行，可以重新执行evaluation.py文件（会跳过已经存在的case，补充缺少的）
