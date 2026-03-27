COCO_META = [
    # --- Things (物品) - 保持鲜艳或略微调整 ---
    {'color': [255, 50, 50], 'isthing': 1, 'id': 1, 'name': 'person'},  # 更亮的红
    {'color': [200, 40, 40], 'isthing': 1, 'id': 2, 'name': 'bicycle'},
    {'color': [30, 144, 255], 'isthing': 1, 'id': 3, 'name': 'car'},  # 道奇蓝
    {'color': [50, 50, 255], 'isthing': 1, 'id': 4, 'name': 'motorcycle'},
    {'color': [147, 112, 219], 'isthing': 1, 'id': 5, 'name': 'airplane'}, # 中紫色
    {'color': [0, 191, 255], 'isthing': 1, 'id': 6, 'name': 'bus'}, # 深天蓝
    {'color': [0, 139, 139], 'isthing': 1, 'id': 7, 'name': 'train'}, # 青色
    {'color': [65, 105, 225], 'isthing': 1, 'id': 8, 'name': 'truck'}, # 皇家蓝
    {'color': [32, 178, 170], 'isthing': 1, 'id': 9, 'name': 'boat'}, # 浅海洋绿
    {'color': [255, 215, 0], 'isthing': 1, 'id': 10, 'name': 'traffic light'}, # 金色
    {'color': [173, 255, 47], 'isthing': 1, 'id': 11, 'name': 'fire hydrant'}, # 黄绿色
    {'color': [255, 255, 0], 'isthing': 1, 'id': 13, 'name': 'stop sign'}, # 纯黄
    {'color': [255, 105, 180], 'isthing': 1, 'id': 14, 'name': 'parking meter'}, # 热粉
    {'color': [255, 69, 0], 'isthing': 1, 'id': 15, 'name': 'bench'}, # 红橙
    {'color': [218, 112, 214], 'isthing': 1, 'id': 16, 'name': 'bird'}, # 兰花紫
    {'color': [255, 0, 255], 'isthing': 1, 'id': 17, 'name': 'cat'}, # 品红
    {'color': [0, 255, 255], 'isthing': 1, 'id': 18, 'name': 'dog'}, # 青色
    {'color': [135, 206, 235], 'isthing': 1, 'id': 19, 'name': 'horse'}, # 天蓝
    {'color': [50, 205, 50], 'isthing': 1, 'id': 20, 'name': 'sheep'}, # 酸橙绿
    {'color': [0, 255, 127], 'isthing': 1, 'id': 21, 'name': 'cow'}, # 春绿
    {'color': [154, 205, 50], 'isthing': 1, 'id': 22, 'name': 'elephant'}, # 黄绿
    {'color': [186, 85, 211], 'isthing': 1, 'id': 23, 'name': 'bear'}, # 中兰花紫
    {'color': [255, 165, 0], 'isthing': 1, 'id': 24, 'name': 'zebra'}, # 橙色
    {'color': [138, 43, 226], 'isthing': 1, 'id': 25, 'name': 'giraffe'}, # 蓝紫
    {'color': [255, 20, 147], 'isthing': 1, 'id': 27, 'name': 'backpack'}, # 深粉
    {'color': [0, 250, 154], 'isthing': 1, 'id': 28, 'name': 'umbrella'}, # 中春绿
    {'color': [255, 0, 127], 'isthing': 1, 'id': 31, 'name': 'handbag'},
    {'color': [123, 104, 238], 'isthing': 1, 'id': 32, 'name': 'tie'}, # 中板岩蓝
    {'color': [64, 224, 208], 'isthing': 1, 'id': 33, 'name': 'suitcase'}, # 绿松石
    {'color': [255, 105, 180], 'isthing': 1, 'id': 34, 'name': 'frisbee'},
    {'color': [199, 21, 133], 'isthing': 1, 'id': 35, 'name': 'skis'}, # 中紫罗兰红
    {'color': [135, 206, 250], 'isthing': 1, 'id': 36, 'name': 'snowboard'}, # 淡天蓝
    {'color': [30, 144, 255], 'isthing': 1, 'id': 37, 'name': 'sports ball'},
    {'color': [127, 255, 0], 'isthing': 1, 'id': 38, 'name': 'kite'}, # 查特酒绿
    {'color': [0, 255, 255], 'isthing': 1, 'id': 39, 'name': 'baseball bat'},
    {'color': [65, 105, 225], 'isthing': 1, 'id': 40, 'name': 'baseball glove'},
    {'color': [210, 180, 140], 'isthing': 1, 'id': 41, 'name': 'skateboard'}, # 棕褐色
    {'color': [176, 196, 222], 'isthing': 1, 'id': 42, 'name': 'surfboard'}, # 淡钢蓝
    {'color': [255, 127, 80], 'isthing': 1, 'id': 43, 'name': 'tennis racket'}, # 珊瑚色
    {'color': [175, 238, 238], 'isthing': 1, 'id': 44, 'name': 'bottle'}, # 苍绿松石
    {'color': [255, 222, 173], 'isthing': 1, 'id': 46, 'name': 'wine glass'}, # 纳瓦霍白
    {'color': [255, 99, 71], 'isthing': 1, 'id': 47, 'name': 'cup'}, # 番茄红
    {'color': [238, 130, 238], 'isthing': 1, 'id': 48, 'name': 'fork'}, # 紫罗兰
    {'color': [192, 192, 192], 'isthing': 1, 'id': 49, 'name': 'knife'}, # 银色
    {'color': [169, 169, 169], 'isthing': 1, 'id': 50, 'name': 'spoon'}, # 暗灰
    {'color': [222, 184, 135], 'isthing': 1, 'id': 51, 'name': 'bowl'}, # 伯莱木
    {'color': [255, 255, 0], 'isthing': 1, 'id': 52, 'name': 'banana'},
    {'color': [255, 0, 0], 'isthing': 1, 'id': 53, 'name': 'apple'},
    {'color': [244, 164, 96], 'isthing': 1, 'id': 54, 'name': 'sandwich'}, # 沙棕色
    {'color': [255, 165, 0], 'isthing': 1, 'id': 55, 'name': 'orange'},
    {'color': [34, 139, 34], 'isthing': 1, 'id': 56, 'name': 'broccoli'}, # 森林绿
    {'color': [255, 140, 0], 'isthing': 1, 'id': 57, 'name': 'carrot'}, # 深橙
    {'color': [205, 92, 92], 'isthing': 1, 'id': 58, 'name': 'hot dog'}, # 印度红
    {'color': [218, 165, 32], 'isthing': 1, 'id': 59, 'name': 'pizza'}, # 金麒麟色
    {'color': [255, 240, 245], 'isthing': 1, 'id': 60, 'name': 'donut'}, # 薰衣草腮红
    {'color': [255, 192, 203], 'isthing': 1, 'id': 61, 'name': 'cake'}, # 粉红
    {'color': [205, 133, 63], 'isthing': 1, 'id': 62, 'name': 'chair'}, # 秘鲁色
    {'color': [70, 130, 180], 'isthing': 1, 'id': 63, 'name': 'couch'}, # 钢蓝
    {'color': [154, 205, 50], 'isthing': 1, 'id': 64, 'name': 'potted plant'},
    {'color': [148, 0, 211], 'isthing': 1, 'id': 65, 'name': 'bed'}, # 深紫
    {'color': [139, 69, 19], 'isthing': 1, 'id': 67, 'name': 'dining table'}, # 马鞍棕
    {'color': [255, 255, 240], 'isthing': 1, 'id': 70, 'name': 'toilet'}, # 象牙白
    {'color': [160, 82, 45], 'isthing': 1, 'id': 72, 'name': 'tv'}, # 赭色
    {'color': [112, 128, 144], 'isthing': 1, 'id': 73, 'name': 'laptop'}, # 板岩灰
    {'color': [190, 190, 190], 'isthing': 1, 'id': 74, 'name': 'mouse'},
    {'color': [128, 128, 128], 'isthing': 1, 'id': 75, 'name': 'remote'},
    {'color': [47, 79, 79], 'isthing': 1, 'id': 76, 'name': 'keyboard'}, # 深板岩灰
    {'color': [255, 0, 255], 'isthing': 1, 'id': 77, 'name': 'cell phone'},
    {'color': [0, 255, 127], 'isthing': 1, 'id': 78, 'name': 'microwave'},
    {'color': [255, 69, 0], 'isthing': 1, 'id': 79, 'name': 'oven'},
    {'color': [255, 215, 0], 'isthing': 1, 'id': 80, 'name': 'toaster'},
    {'color': [224, 255, 255], 'isthing': 1, 'id': 81, 'name': 'sink'}, # 淡青
    {'color': [240, 255, 255], 'isthing': 1, 'id': 82, 'name': 'refrigerator'}, # 天蓝色
    {'color': [210, 105, 30], 'isthing': 1, 'id': 84, 'name': 'book'}, # 巧克力色
    {'color': [255, 255, 0], 'isthing': 1, 'id': 85, 'name': 'clock'},
    {'color': [255, 160, 122], 'isthing': 1, 'id': 86, 'name': 'vase'}, # 浅鲑鱼色
    {'color': [255, 0, 170], 'isthing': 1, 'id': 87, 'name': 'scissors'},
    {'color': [255, 192, 203], 'isthing': 1, 'id': 88, 'name': 'teddy bear'},
    {'color': [255, 20, 147], 'isthing': 1, 'id': 89, 'name': 'hair drier'},
    {'color': [176, 224, 230], 'isthing': 1, 'id': 90, 'name': 'toothbrush'}, # 粉蓝

    # --- Stuff (背景/环境) - 重点修改区域 ---
    {'color': [255, 255, 100], 'isthing': 0, 'id': 92, 'name': 'banner'},
    {'color': [173, 216, 230], 'isthing': 0, 'id': 93, 'name': 'blanket'}, # 淡蓝
    {'color': [205, 133, 63], 'isthing': 0, 'id': 95, 'name': 'bridge'}, # 提亮
    {'color': [222, 184, 135], 'isthing': 0, 'id': 100, 'name': 'cardboard'},
    {'color': [210, 180, 140], 'isthing': 0, 'id': 107, 'name': 'counter'},
    {'color': [255, 228, 196], 'isthing': 0, 'id': 109, 'name': 'curtain'}, # 浓汤乳黄
    {'color': [160, 82, 45], 'isthing': 0, 'id': 112, 'name': 'door-stuff'},
    {'color': [244, 164, 96], 'isthing': 0, 'id': 118, 'name': 'floor-wood'}, # 提亮
    {'color': [255, 105, 180], 'isthing': 0, 'id': 119, 'name': 'flower'},
    {'color': [255, 0, 255], 'isthing': 0, 'id': 122, 'name': 'fruit'},
    {'color': [169, 169, 169], 'isthing': 0, 'id': 125, 'name': 'gravel'},
    # 🔥 重点修改：house (房子) - 改为鲜艳的青色
    {'color': [0, 255, 255], 'isthing': 0, 'id': 128, 'name': 'house'},
    {'color': [255, 255, 224], 'isthing': 0, 'id': 130, 'name': 'light'},
    {'color': [240, 255, 240], 'isthing': 0, 'id': 133, 'name': 'mirror-stuff'}, # 蜜瓜绿
    {'color': [255, 99, 71], 'isthing': 0, 'id': 138, 'name': 'net'},
    {'color': [255, 182, 193], 'isthing': 0, 'id': 141, 'name': 'pillow'}, # 浅粉
    {'color': [255, 218, 185], 'isthing': 0, 'id': 144, 'name': 'platform'}, # 桃色
    {'color': [152, 251, 152], 'isthing': 0, 'id': 145, 'name': 'playingfield'},
    {'color': [188, 143, 143], 'isthing': 0, 'id': 147, 'name': 'railroad'}, # 玫瑰棕
    {'color': [0, 191, 255], 'isthing': 0, 'id': 148, 'name': 'river'}, # 提亮
    # 🔥 重点修改：road (道路) - 提亮紫色饱和度
    {'color': [180, 100, 180], 'isthing': 0, 'id': 149, 'name': 'road'},
    # 🔥 重点修改：roof (屋顶) - 改为鲜艳的橙红
    {'color': [255, 90, 0], 'isthing': 0, 'id': 151, 'name': 'roof'},
    {'color': [238, 232, 170], 'isthing': 0, 'id': 154, 'name': 'sand'}, # 苍麒麟色
    {'color': [64, 224, 208], 'isthing': 0, 'id': 155, 'name': 'sea'}, # 提亮
    {'color': [255, 160, 122], 'isthing': 0, 'id': 156, 'name': 'shelf'},
    {'color': [250, 250, 255], 'isthing': 0, 'id': 159, 'name': 'snow'},
    {'color': [210, 180, 140], 'isthing': 0, 'id': 161, 'name': 'stairs'},
    {'color': [255, 235, 205], 'isthing': 0, 'id': 166, 'name': 'tent'}, # 白杏色
    {'color': [255, 240, 245], 'isthing': 0, 'id': 168, 'name': 'towel'},
    # 🔥 重点修改：wall-brick (砖墙) - 改为鲜艳的砖红
    {'color': [230, 60, 60], 'isthing': 0, 'id': 171, 'name': 'wall-brick'},
    # 🔥 重点修改：wall-stone (石墙) - 改为鲜艳的亮蓝灰
    {'color': [100, 150, 255], 'isthing': 0, 'id': 175, 'name': 'wall-stone'},
    # 🔥 重点修改：wall-tile (瓷砖墙) - 保持高亮青色
    {'color': [0, 255, 230], 'isthing': 0, 'id': 176, 'name': 'wall-tile'},
    # 🔥 重点修改：wall-wood (木墙) - 改为鲜艳的金棕色
    {'color': [220, 170, 50], 'isthing': 0, 'id': 177, 'name': 'wall-wood'},
    {'color': [0, 200, 255], 'isthing': 0, 'id': 178, 'name': 'water-other'},
    {'color': [255, 192, 203], 'isthing': 0, 'id': 180, 'name': 'window-blind'},
    {'color': [255, 100, 120], 'isthing': 0, 'id': 181, 'name': 'window-other'}, # 提亮
    # 🔥 重点修改：tree-merged (树) - 改为鲜艳的森林绿
    {'color': [34, 200, 34], 'isthing': 0, 'id': 184, 'name': 'tree-merged'},
    {'color': [233, 150, 122], 'isthing': 0, 'id': 185, 'name': 'fence-merged'}, # 提亮深鲑鱼色
    {'color': [245, 245, 220], 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'}, # 米色
    # 🔥 重点修改：sky-other-merged (天空) - 改为鲜艳的天蓝色
    {'color': [135, 206, 235], 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'},
    {'color': [255, 218, 185], 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'},
    {'color': [255, 250, 205], 'isthing': 0, 'id': 189, 'name': 'table-merged'}, # 柠檬绸色
    {'color': [221, 160, 221], 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'}, # 梅红色
    # 🔥 重点修改：pavement-merged (路面) - 改为极亮的浅灰色
    {'color': [230, 230, 230], 'isthing': 0, 'id': 191, 'name': 'pavement-merged'},
    {'color': [143, 188, 143], 'isthing': 0, 'id': 192, 'name': 'mountain-merged'}, # 海洋绿
    # 🔥 重点修改：grass-merged (草地) - 改为鲜艳的酸橙绿
    {'color': [124, 252, 0], 'isthing': 0, 'id': 193, 'name': 'grass-merged'},
    {'color': [210, 105, 30], 'isthing': 0, 'id': 194, 'name': 'dirt-merged'}, # 巧克力色
    {'color': [255, 255, 240], 'isthing': 0, 'id': 195, 'name': 'paper-merged'},
    {'color': [255, 69, 0], 'isthing': 0, 'id': 196, 'name': 'food-other-merged'},
    # 🔥 重点修改：building-other-merged (建筑) - 改为鲜艳的金色
    {'color': [255, 215, 0], 'isthing': 0, 'id': 197, 'name': 'building-other-merged'},
    {'color': [169, 169, 169], 'isthing': 0, 'id': 198, 'name': 'rock-merged'},
    # 🔥 重点修改：wall-other-merged (其他墙) - 改为鲜艳的品红/紫
    {'color': [255, 50, 255], 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'},
    {'color': [255, 182, 193], 'isthing': 0, 'id': 200, 'name': 'rug-merged'}
]