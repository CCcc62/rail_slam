# 项目名称：面向铁路场景的语义地图关键技术研究

## 项目简介
本项目实现了基于视觉 SLAM 的铁路场景三维语义地图构建，包括点云生成、相机位姿估计、电线杆点云聚类与可视化等功能。主要代码包括 `slam.py`（SLAM主流程）、`draw_pcd.py`（点云与电线杆可视化）、以及相关数据文件。

## 目录结构
- `slam.py`：主 SLAM 流程，处理视频帧，生成点云和相机位姿。
- `draw_pcd.py`：点云、电线杆、相机轨迹的可视化。
- `CRH_1.txt`、`movie_003.txt`：检测框、特征点等原始数据。
- `pcd.txt`、`pcd_bbox.txt`：生成的三维点云数据、电线杆点云。
- `pose.txt`：相机位姿数据。
- `processed.txt`：处理后的电线杆点云中心点。

## 依赖环境
- Python 3.x
- OpenCV
- Pangolin
- Numpy
- Pandas
- scikit-image
- OpenGL

## 使用方法

1. **准备数据**  
   将原始视频和检测框数据放入项目目录，确保路径与代码一致。

2. **运行 SLAM 主流程**  
   生成点云和相机位姿：
   python slam.py
   结果会输出到 `pcd.txt`、`pcd_bbox.txt`、`pose.txt`。

3. **点云与电线杆可视化**  
   可视化三维地图和电线杆：
   python draw_pcd.py

4. **参数调整**  
   可根据实际视频分辨率和相机参数修改 `W`, `H`, `F` 等参数。

## 数据格式说明

- `CRH_1.txt`、`movie_003.txt`：每行表示一个检测框或特征点，格式为 `帧号 类别 x y w h ...`
- `pcd.txt`：每行一个三维点，格式为 `x y z 1`
- `pcd_bbox.txt`：每行一个电线杆点云，格式为 `类别 x y z 1`
- `pose.txt`：每 4 行一个 4x4 位姿矩阵

## 可视化效果
- 绿色：相机轨迹
- 灰色：普通点云
- 紫色：检测框点云
- 蓝色：电线杆点云

## 参考
- 视觉 SLAM 相关论文
- OpenCV 官方文档
- Pangolin 官方文档

## 联系方式
如有任何文件或建议，请通过Issue或者邮件联系
