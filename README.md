# resnet-for-single-channel
Usually, resnet accept three chnnels image input, but in some cases, we want keeping the pretrained model(both the net and parameters), so this code aims at this case. Accept single channel input, then using the pretrained resnet model to inference graysclae image.

dependecny:
1.python3.7

How to run it.
1. pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

