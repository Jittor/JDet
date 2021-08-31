# 模型测试
## 测试你的模型是否对其他模型产生影响
在你完成一个模型{model}之后，请先测试完成的模型会不会对其他模型的正确性产生影响，如果所有模型都输出"success"则为成功：
```sh
python tools/test_models.py
```

`test_models.py`会自动从网盘下载每个模型的输入数据及若干迭代的loss，通过对比现在每个模型在相同输入情况下的loss与保存的loss的异同来检查每个模型的正确性。

## 添加你的模型的测试
之后为了避免其他模型编写过程中对你实现的模型产生影响，你需要对实现的模型增加一个测试脚本。

测试脚本的功能是：当你训练这个模型达到预期点数之后，说明这个版本的代码是正确的，这时在输入固定的情况下，把训练若干iter的loss保存起来，在之后的版本即可通过对比同样输入下若干iter的loss来检查你的模型的正确性。

## 如何编写测试脚本
1. `创建projects/{model}/test_{model}.py`，内容请参考`projects/retinanet/test_retinanet.py`。
2. test_{model}.py包含两种模式set_data模式和测试模式，如果是set_data模式会将网络若干迭代的输入及loss存在`projects/{model}/test_datas_{model}`中。
3. 如果是测试模式，会自动导入保存的输入，并对比若干迭代的loss与保存的loss。
4. <b>你需要为测试准备一个独立的config文件，关闭一些可能产生随机的操作。为保证测试能体现反向传播的正确性，可以尽量调大learning rate。</b>
5. 多跑几次，把loss打出来看看是否稳定，如果不稳定可能是什么随机操作还没有完全关闭。
6. 请保证在有文件`projects/{model}/test_datas_{model}`的情况下直接通过脚本`python test_{model}.py`即可进行测试，保证如果某个iter的loss差别过大则返回错误，如果通过测试则打印"success"。
7. 如果网络需要用到什么预训练文件，也可以放到`projects/{model}/test_datas_{model}`中，并把test的config文件中预训练文件的路径改为`test_datas_{model}/xx.pk`（可以参考retinanet），这是为了保证其他用户可以在不需要下载其他文件的情况下，直接通过`python test_{model}.py`进行测试。
8. 在编写完成后可以多跑几次`python test_{model}.py`，给loss的错误率寻找一个合适的阈值。

## 把你的测试加入到test_models.py脚本中
1. 在`tools/test_models.py`第二行的tasks中添加{model}。
2. 把projects/{model}/test_datas_{model}拷贝到test_datas文件夹中，文件格式为
```
test_datas
├── test_datas_retinanet
|     ├──test_data.pk
|     └──yx_init_pretrained.pk_jt.pk
├── ...
|     ├──...
|     └──...
└── test_datas_{model}
      ├──test_data.pk
      └──...

```
3. 压缩`zip -r test_datas.zip test_datas`。
4. 把test_datas.zip传到清华网盘，共享并获取直接下载链接。
5. 把直接下载链接填入`tools/test_models.py`第三行的zip_path。
6. 可以删除掉test_datas.zip然后运行`python tools/test_models.py`测试一下正确性。