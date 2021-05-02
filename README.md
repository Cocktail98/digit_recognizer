# digit_recognizer

Practice deep learning with MNIST dataset

以 Kaggle上面 MNIST 数据集的比赛[`Digit Recognizer`](https://www.kaggle.com/c/digit-recognizer)作为练习写的两个小程序，分别使用[`Python PyTorch`](./py_pytorch/)与[`C++ TensorRT`](./cpp_tensorRT)实现。作为熟悉这两个框架的小练习，难度较低，非常值得一看。

其中Python的Pyorch实现的使用方法就直接运行main.py文件就好，需要注意需要自己更改项目中的参数，例如train还是test模式，epoch还是其他的一些参数。

TensorRT的程序需要您首先训练出来一个模型参数，然后使用[`gen_wts.py`](./py_pytorch/gen_wts.py)文件生成对应的网络参数`cnn.wts`（注意可能需要更改程序中的路径问题），若不更改，则默认将`cnn.wts`文件存放于`./cpp_tensorRT/`文件夹下面。

```shell
cp {file_dir}/cnn.wts {digit_recognizer}/cpp_tensorRT/cnn.wts
```

然后您需要cd到TensorRT程序的文件夹下，然后构建模型，之后运行：

```shell
cd cpp_tensorRT/
mkdir build
cd build/
cmake ..
make
./main -s # 构建模型
./main -d # 进行推理
```

最后会生成一个`submission.csv`文件，可以将其上[Kaggle](https://www.kaggle.com/c/digit-recognizer/submit)上面提交哦。

Inspired by [wang-xinyu](https://github.com/wang-xinyu/tensorrtx)
