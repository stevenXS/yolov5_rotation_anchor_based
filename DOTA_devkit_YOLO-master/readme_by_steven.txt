DOTA数据裁剪教程：

1.运行ImgSplit.py(多线程也可以) , 此时得到裁剪后的标签和图片

2.如果有检测结果的TXT文件，可以运行ResultMerge.py将结果集合进行合并（可选）

3.运行YOLO_Transform.py，将DOTA的格式转化为YOLO （注，默认是归一化的坐标给训练使用，如果需要调用COCOApi评估，则使用非归一化版本）