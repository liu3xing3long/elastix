#!/bin/bash
python setup.py build
python setup.py install
python test.py>log.log
