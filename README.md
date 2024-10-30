# TouchGUI

## 简介

本项目基于python的ultralytics、pyautogui、pyinstaller库开发，用于识别并滑动弹窗



流程：使用ultralytics进行方便的模型训练和验证->挑选合适模型后使用独立模型框架，减少引用库->使用pyautogui完成交互框架->pyinstaller打包所需文件



## 前期模型训练

作者挑选pytorch(pt)，onnx，openVION三种模型格式，使用ultralytics库中完善的函数来进行训练测试。

训练过程YOLO官方仅支持pt作为模型输出格式，其他模型均由pt模型转化过来。

训练代码：

```python
from ultralytics import YOLO

model = YOLO("models/yolov8s.pt") # build a new model

# Train the model
results = model.train(data="puzzleBlockDatasets.yaml", cfg="trainCfg.yaml")
```

转换代码（本项目转换模型格式在命令行执行）：

```shell
yolo export model=.//models//puzzle_s.pt format=onnx
yolo export model=.//models//puzzle_s.pt format=openvion int8=True dynamic=True
yolo export model=.//models//puzzle_s.pt format=openvion dynamic=True
```

## 测试模型

使用jupyter notebook的%timeit工具，对模型进行多重时间测试，以下是测试表单，经测试，openVION和openVION_int8模型在某些特定情况下会卡死，已放弃作为候选

>  测试集为161张含有滑块拼图的图，测试场景为CPU推理，CPU配置为i5-12500H。

| model        | time            | speed    | P    | R     |
| ------------ | --------------- | -------- | ---- | ----- |
| ONNX         | 34.6 s ± 30.5 s | 约2it/s  | 1    | 1     |
| pytorch      | 10.1 s ± 115 ms | 1.25it/s | 1    | 1     |
| openVION     | /               | 7.79it/s | 1    | 1     |
| openVION_int | /               | 5.35it/s | 1    | 0.969 |

测试下来，发现原生pt的速度最快，误差最小，因此决定使用pytorch格式作为模型格式。

测试代码：

```python
from ultralytics import YOLO
datasetYaml = ".//config//puzzleBlockDatasets.yaml"
def test():
    # Run batched inference on a list of images
    results = model.val(data=datasetYaml,device='cpu',conf=0.8,max_det=1)
%timeit test()
```

## 模型框架



## 交互框架



## 打包可执行文件



## 参考文章

[Ultralytics YOLO Docs](https://docs.ultralytics.com/)

[MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights)

[Python 进阶必学库：Pyinstaller 使用详解 ！](https://zhuanlan.zhihu.com/p/71081512)

[PyAutoGui 图片识别+定位+截图函数文档](https://www.cnblogs.com/math98/p/14399644.html)

[pyAutoGUI Docs](https://pyautogui.readthedocs.io/en/latest/index.html)

[Python获取当前时间（time模块）](https://blog.csdn.net/qq_36512295/article/details/99694528)

[OpenCV 图像缩放：cv.resize() 函数详解](https://blog.csdn.net/hysterisis/article/details/112381220)

[OpenCV-Python实战(1) —— 给图片添加图片水印【利用 OpenCV 像素的读写原理实现】](https://blog.csdn.net/m0_38082783/article/details/127445270)
# Touch
