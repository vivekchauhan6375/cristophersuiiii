#!/bin/bash

sudo apt install python3-pip

sudo apt install python3-venv

python3 -m .venv .venv

source .venv/bin/activate

code .
 
pip3 install -r requirnments.txt