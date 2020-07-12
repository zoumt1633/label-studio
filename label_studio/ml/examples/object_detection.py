# -*- coding: utf-8 -*-
# @Time    : 2020/7/12 14:19
# @Author  : zoumaotai
# @Email   : zoumaotai@ailongma.com
# @File    : object_detection.py
# @Software: PyCharm

import random
import urllib
from gluoncv import model_zoo, data
from label_studio.ml import LabelStudioMLBase
import mxnet as mx


class ObjectDetectionModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(ObjectDetectionModel, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']
        self.net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            image_url = task.get('data').get('image')
            image_url = f'http://localhost:8080{image_url}' if not image_url.startswith('http') else image_url
            urllib.request.urlretrieve(image_url, "test.jpg")
            src = mx.img.imread('test.jpg')
            org_h, org_w, _ = src.shape
            x, orig_img = data.transforms.presets.rcnn.load_test("test.jpg")
            h, w, _ = orig_img.shape
            ratio_h = org_h/h
            ratio_w = org_w/w
            print('h缩放比例', ratio_h)
            print('w缩放比例', ratio_w)
            box_ids, scores, bboxes = self.net(x)
            result_list = []
            for bbox, box_id, score in zip(bboxes[0].asnumpy().tolist(), box_ids[0].asnumpy().tolist(),
                                           scores[0].asnumpy().tolist()):
                if bbox[0] == -1:
                    break
                label = self.net.classes[int(box_id[0])]
                score = score[0]
                x = bbox[0] * ratio_w * 100 / org_w
                y = bbox[1] * ratio_h * 100 / org_h
                height = (bbox[3] - bbox[1]) * ratio_h * 100 / org_h
                width = (bbox[2] - bbox[0]) * ratio_w * 100 / org_w
                if score > 0.8:
                    result_list.append(
                        {
                            "from_name": "label",
                            "id": "t5sp3TyXPo",
                            "source": "$image",
                            "to_name": "image",
                            "type": "rectanglelabels",
                            "value": {
                                "height": height,   # 高度占比
                                "rectanglelabels": [
                                    label
                                ],
                                "rotation": 0,
                                "width": width,   # 宽度占比
                                "x": x,
                                "y": y
                            }
                        })
            results.append({
                "result": result_list,
                'score': 0.9
            })
        print(results)
        return results

    def fit(self, completions, workdir=None, **kwargs):
        return {'random': random.randint(1, 10)}

