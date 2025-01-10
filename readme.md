# ruc2024秋季学期深度学习大作业项目配置文档

## 正常配置

环境配置：[环境配置文件](environment.yml)

数据集构建请参考[mmseg](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md)中关于cocostuff164K的搭建方式。

搭建好后请运行：[cocoob/cvt_coco_object.py](cocoob/cvt_coco_object.py) 将数据集转为cocoob数据集，放在cocoob文件夹下。

运行请执行：

```bash
python eval.py
```

运行前或者遇到错误，记得检查[cfg_coco_object.py](configs/cfg_coco_object.py)和[base_config.py](configs/base_config.py)中的路径是否有错误。这两个文件已经由我配置好。

灵活运用nohup 和 后台运行以节省精力。

## 快速上手

请进行环境配置之后，直接运行[demo](demo.py)

```bash
python demo.py
```

注意修改demo.py第六行的**图片路径**以及**文本列表**以适应新的图片分割任务。

demo的结果会输出在./image 文件夹下。

## 网页部分

[index.html](index.html)为主页面。

如需体验，请在**配置好环境**之后在终端中运行：

```bash
ngrok http http://localhost:8080
```

将运行后显示的类似`https://1bac-36-112-3-77.ngrok-free.app`的网址替换掉index.html第160行的网址。

重新开一个终端，运行：

```bash
python app.py
```

然后点击新的类似`https://1bac-36-112-3-77.ngrok-free.app`的网址，或者在其他电脑上访问该网址，就能实现前端传递图片到本机，本机处理后返回处理结果的功能。

为方便查看结果，我录制了[视频演示](web演示.mp4)
