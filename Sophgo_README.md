
开始训练

## eager mode训练
```shell
export DISABLE_CACHE=1
export TorchTpuSaveKernelModule=1
python train.py --batch-size 8 --device tpu
```

## 编译模式训练
```shell
export DISABLE_CACHE=1
export TorchTpuSaveKernelModule=1
python train.py --batch-size 8 --device tpu --compile --compiler aot
```

## DEBUG Features： generate fx-graph and reference io
```shell
python train.py --batch-size 8 --device cpu --compile --compiler dummy
```