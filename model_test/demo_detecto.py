# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 23:24
# @Author  : zoumaotai
# @Email   : zoumaotai@ailongma.com
# @File    : demo_detecto.py
# @Software: PyCharm

from detecto.core import Model
from detecto import utils

print('初始化模型')
model = Model()
print('初始化结束')

image = utils.read_image('test.jpg')
predict = model.predict(image)
print(predict)
