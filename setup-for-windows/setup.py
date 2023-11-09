#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import codecs
import sys
import os
import re
from setuptools import setup, find_packages

setup(
    name='parl',
    version=1.1,
    description='Reinforcement Learning Framework',
    url='https://github.com/PaddlePaddle/PARL',
	packages=[package for package in find_packages()
              if package.startswith('parl')],
    package_data={'': ['*.so']},
    install_requires=[
        "termcolor>=1.1.0",
        "pyzmq==18.0.1",
        "pyarrow==14.0.1",
        "scipy>=1.0.0",
        "cloudpickle==1.0.0",
        "tensorboardX",
        "tensorboard",
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
