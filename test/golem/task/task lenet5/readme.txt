1、input文件夹里有数字0-9的png和bin格式的文件，大小是28*28*4byte，数据都是fp32格式的
2、lenet5.onnx是onnx格式的模型文件
3、lenet5.onnx.png是通过Netron可视化工具输出得到的模型计算图
4、lenet5_fp32.weights是lenet5模型的权重数据，按顺序保存了各个层的权重偏置参数，数据都是fp32格式的
5、lenet计算过程(拆开卷积).png是把卷积拆开成im2col+gemm的格式
6、lenet5_c工程主要工作是加载模型参数和输入数据，执行lenet5推理过程，可部署在vanadis上（注意需要修改模型参数和输入数据的路径）
执行命令如下：make all
             ./lenet5 a(a为执行线程数，例如./lenet5 2)
            