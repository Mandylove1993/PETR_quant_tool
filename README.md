# PETR QUANT DEMO TOOL

This repository is a quant implementation demo of [PETR](https://arxiv.org/abs/2203.05625).
Just for self study, I use the official "petr_vov_p4_800x320.pth" for development.


# Usage
## PTQ

You can PTQ the model following:
```bash
python tools/quant/ptq_bev.py
```

## QAT

You can QAT the model following:
```bash
python tools/quant/qat_bev.py 
```
## export onnx
You can generate the onnx following:
```bash
python tools/quant/export_onnx.py
```


## Main Results

|config	               | mAP     |	NDS      	|Latency|
|:--------:|:----------:|:---------:|:--------:|
PETR-vov-p4-800x320	|   37.8%	|   42.6%	  | 64.8768 ms|
PTQ                |    32.89%	 |  30.20%	 |  31.5722 ms|
QAT	               |    30.94%	|   27.82%	||

* QAT Due to limited personal resources, a single card 3080 single batch trained for 10 epochs, mAP has been rising, there should be room for improvement
* onnx and pth will be pushed later

## Acknowledgement
Many thanks to the authors of [Lidar_AI_Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution) and [PETR](https://github.com/megvii-research/PETR) .

## Contact
If you have any questions, feel free to open an issue.
