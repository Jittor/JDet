## JDet baseline使用说明

### 1. 安装JDet

进入JDet的github主页：https://github.com/Jittor/JDet

按照README的提示安装JDet及其所需环境，安装完成后切换到JDet-competition分支。

### 2. 下载数据集&数据预处理

1. 下载数据集，解压后修改成以下形式：

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

