from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
# from datasets.coco import coco

import numpy as np

for year in ['2007','2012']:
    for split in ['train','val','trainval','test']:
        name = 'voc_{}_{}'.format(year,split)
        __sets[name] = (lambda split=split,year=year:pascal_voc(split,year))

for year in ['2007','2012']:
    for split in ['train','val','trainval','test']:
        name = 'voc_{}_{}_diff'.format(year,split)
        __sets[name] = (lambda split=split,year=year:pascal_voc(split,year,use_diff=True))

# for year in ['2014']:
#     for split in ['train','val','minival','valminusminival','trainval']:
#         name = 'coco_{}_{}'.format(year,split)
#         __sets[name] = (lambda split=split,year=year:coco(split,year))
#
# for year in ['2015']:
#     for split in ['test','test-dev']:
#         name = 'coco_{}_{}'.format(year,split)
#         __sets[name] = (lambda split=split,year=year:coco(split,year))

def get_imdb(name):
    # 根据名字获取image database
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    # 列出所有登记过的图片库
    return list(__sets.keys())
