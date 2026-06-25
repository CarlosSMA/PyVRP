#!/usr/bin/env/bash

pip install -r ./requirements.txt
pip install -e .

cd ./execution

python3 ./main.py