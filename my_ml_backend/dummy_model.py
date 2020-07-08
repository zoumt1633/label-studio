import random

from label_studio.ml import LabelStudioMLBase


class DummyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        super(DummyModel, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks, **kwargs):
        results = []
        for task in tasks:
            results.append({
                "result": [
                    {
                        "from_name": "label",
                        "id": "t5sp3TyXPo",
                        "source": "$image",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {
                            "height": 11.6122840691,
                            "rectanglelabels": [
                                "Airplane"
                            ],
                            "rotation": 0,
                            "width": 39.6,
                            "x": 13.2,
                            "y": 34.7024952015
                        }
                    }
                ],
                'score': random.uniform(0, 1)
            })
        return results

    def fit(self, completions, workdir=None, **kwargs):
        return {'random': random.randint(1, 10)}
