
numpy=1.24.3
amp
eval arang f16 问题，
f16 maxpool2d, hang.

开始训练
```shell
export DISABLE_CACHE=1
export TorchTpuSaveKernelModule=1
python train.py --batch-size 8 --device tpu --compile --noval
```

```
python val.py --weights yolov5s.pt --data coco128.yaml --img 640  --batch 8
```