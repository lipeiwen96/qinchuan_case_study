# src/QINCHUAN_LABELS_MAP.py
# -*- coding: utf-8 -*-

"""
芹川古村专用视觉对象标签体系 (优化版 - 涵盖电线杆、广告牌及红砖房)
键(Key): 发送给 CLIP 的英文描述 (Prompt)
值(Value): 最终显示的中文类别标签 (共10类)
"""

QINCHUAN_LABELS_MAP = {
    # ==========================================
    # 1. 传统屋顶 / 木构立面 (Heritage: Roof & Wood)
    # ==========================================
    "traditional tiled roof of a Chinese historic house": "1.传统屋顶/木构",
    "grey clay tiled roof with flying eaves": "1.传统屋顶/木构",
    "aged timber facade and wooden walls": "1.传统屋顶/木构",
    "old wooden pillars and beams": "1.传统屋顶/木构",
    "traditional whitewashed horse-head wall": "1.传统屋顶/木构",  # 徽派马头墙

    # ==========================================
    # 2. 门楼 / 祠堂入口 (Heritage: Gate & Entrance)
    # ==========================================
    "ancestral hall gate tower and entrance": "2.门楼/祠堂入口",
    "traditional Chinese courtyard entrance gate": "2.门楼/祠堂入口",
    "stone gate frame with carvings": "2.门楼/祠堂入口",
    "traditional wooden door of a house": "2.门楼/祠堂入口",

    # ==========================================
    # 3. 窗棂 / 装饰细部 (Heritage: Windows & Details)
    # ==========================================
    "intricate window carvings and architectural details": "3.窗棂/装饰细部",
    "traditional wooden lattice window": "3.窗棂/装饰细部",
    "carved wooden window frames": "3.窗棂/装饰细部",
    "decorative hollow brick window": "3.窗棂/装饰细部",  # 像图4中的花窗

    # ==========================================
    # 4. 传统墙体 / 砖石纹理 (Heritage: Masonry & Texture)
    # ==========================================
    "traditional white masonry wall": "4.传统墙体/纹理",
    "grey brick wall texture": "4.传统墙体/纹理",
    "stone base of the wall": "4.传统墙体/纹理",
    "weathered wall texture": "4.传统墙体/纹理",
    "wall made of irregular stones": "4.传统墙体/纹理",  # 图中常见的乱石墙基

    # ==========================================
    # 5. 植被 / 古树 (Nature: Vegetation)
    # ==========================================
    "ancient tree and mature vegetation": "5.植被/古树",
    "green bushes and plants": "5.植被/古树",
    "potted plants in front of house": "5.植被/古树",
    "ivy or vines on the wall": "5.植被/古树",

    # ==========================================
    # 6. 水系 / 桥 (Nature: Water & Bridge)
    # ==========================================
    "river stream and water canal": "6.水系/桥",
    "stone arch bridge over river": "6.水系/桥",
    "stone steps leading to water": "6.水系/桥",
    "concrete slab bridge": "6.水系/桥",  # 图中也有水泥板桥

    # ==========================================
    # 7. 现代建筑 / 红色面砖 (Modern Intrusion: Building)
    # 针对图97/95/100中的红砖房和现代白瓷砖房
    # ==========================================
    "modern concrete building addition": "7.现代建筑/红砖",
    "building with red brick tile facade": "7.现代建筑/红砖",
    "building with white ceramic tile facade": "7.现代建筑/红砖",
    "modern aluminum window with bars": "7.现代建筑/红砖",  # 防盗窗

    # ==========================================
    # 8. 现代设施 / 视觉干扰 (Modern Intrusion: Objects)
    # 涵盖车辆、电线杆、广告牌、空调
    # ==========================================
    "modern vehicle or car parked in the street": "8.现代设施/干扰",
    "motorcycle or electric scooter": "8.现代设施/干扰",
    "electric wires and utility poles": "8.现代设施/干扰",  # 电线杆
    "messy overhead cables": "8.现代设施/干扰",  # 乱拉电线
    "air conditioner unit on wall": "8.现代设施/干扰",
    "blue plastic shed or temporary structure": "8.现代设施/干扰",
    "signboard or billboard with text": "8.现代设施/干扰",  # 广告牌/标语
    "red couplets or banners on the wall": "8.现代设施/干扰",  # 春联/横幅 (虽有文化属性，但多为现代材质，暂归此类或单独列出)

    # ==========================================
    # 9. 生活杂物 / 人 (Life & Clutter)
    # ==========================================
    "pedestrian or tourist walking": "9.生活杂物/人",
    "group of people": "9.生活杂物/人",
    "pile of trash and visual clutter": "9.生活杂物/人",
    "hanging laundry or clothes": "9.生活杂物/人",  # 晾衣服
    "household items like buckets and brooms": "9.生活杂物/人",  # 水桶扫帚
    "plastic chairs and tables outside": "9.生活杂物/人",  # 门口坐的塑料桌椅

    # ==========================================
    # 10. 天空 / 背景 (Background)
    # ==========================================
    "blue sky and clouds": "99.天空/背景",
    "clear sky": "99.天空/背景",
    "distant mountains": "99.天空/背景",
    "paved stone path ground": "99.天空/背景",  # 地面归入背景，或者单独一类，这里为了凑10类归入背景
    "concrete road surface": "99.天空/背景",

    # 兜底
    "unknown object": "99.天空/背景"
}