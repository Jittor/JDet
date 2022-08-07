## 粤港澳大湾区（黄埔）国际算法算力大赛-遥感图像目标检测赛道-Baseline使用说明

比赛官方基于计图遥感图像目标检测算法库JDet提供了本次比赛的Baseline model，使用说明如下。

### 1. 安装JDet

进入JDet的github主页：https://github.com/Jittor/JDet

按照README的提示安装JDet及其所需环境。

注：Baseline暂时只支持在Ubuntu系统中运行。

### 2. 下载数据集&数据预处理

1. 在比赛页面下载FAIR1M 1.5数据集，解压后修改成以下形式：

    ```
    {DATASET_PATH}
        |
        └──data 
            ├──train
            |     ├──images
            |     |    ├──1.tif
            |     |    └──...
            |     └──labelXml
            |          ├──1.xml
            |          └──...
            └──test
                  └──images
                       ├──1.tif
                       └──...
    ```

    其中`{DATASET_PATH}`为数据集路径，用户可以自行选择。

    **注意：直接解压数据集得到的文件树可能与说明不同（如labelXml、test的名称），请将其修改为说明中的格式。**

2. 由于遥感图像尺寸较大，需要先使用JDet进行数据预处理：

    进入`configs/preprocess/fair1m_1_5_preprocess_config.py`文件，修改这个文件中的三个路径参数为

    ```python
    source_fair_dataset_path='{DATASET_PATH}/data'
    source_dataset_path='{DATASET_PATH}/dota'
    target_dataset_path='{DATASET_PATH}/preprocessed'
    ```

    其中`{DATASET_PATH}`与前一步相同。

    在`JDet`目录下执行`python tools/preprocess.py --config-file configs/preprocess/fair1m_1_5_preprocess_config.py`，即可自动进行数据预处理。

3. baseline训练和测试

    数据预处理完成后，我们就可以对模型进行训练和测试。JDet目前提供基于s2anet的baseline。

    使用方法：修改`configs/s2anet/s2anet_r50_fpn_1x_fair1m_1_5.py`文件，**将`dataset_root`一项设为第一步中的`{DATASET_PATH}`**。修改完成后在`JDet`目录下执行`python tools/run_net.py --config-file configs/s2anet/s2anet_r50_fpn_1x_fair1m_1_5.py`即可自动进行训练和测试。

    测试得到的结果会储存在`submit_zips/s2anet_r50_fpn_1x_fair1m_1_5.csv`，将这个文件提交到竞赛网站上可以进行评测


### 3. 其他资源
为了方便各位选手使用和交流我们的官方Baseline代码和计图框架，我们创建了**遥感图像目标检测赛道选手交流QQ群**，请各位选手通过群号526028350或扫描以下的二维码入群：

<img src="https://user-images.githubusercontent.com/73881739/183240213-5b0ef70f-e11e-4d1c-bb3c-87c4e8a987f3.jpg" alt="QQ_Group" width="400"/>


Jittor 是一个基于即时编译和元算子的高性能深度学习框架，整个框架在即时编译的同时，还集成了强大的Op编译器和调优器，为您的模型生成定制化的高性能代码。Jittor还包含了丰富的高性能模型库，涵盖范围包括：图像识别、检测、分割、生成、可微渲染、几何学习、强化学习等。

Jittor前端语言为Python，使用了主流的包含模块化和动态图执行的接口设计，后端则使用高性能语言进行了深度优化。更多关于Jittor的信息可以参考：
*  [Jittor官网](https://cg.cs.tsinghua.edu.cn/jittor/)
*  [Jittor教程](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
*  [Jittor模型库](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
*  [Jittor文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
*  [Github开源仓库](https://github.com/jittor/jittor)
*  [Jittor论坛](https://discuss.jittor.org)

