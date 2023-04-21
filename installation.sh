#!/usr/bin/env bash

set -x

pip install tensorboard
pip install wandb
pip install gluoncv
# pip install mxnet
sudo cp /opt/conda/lib/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/
sudo rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
sudo ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.29 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
# strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX