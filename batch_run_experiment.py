# batch_run_experiment.py
# -*- coding: utf-8 -*-
import os
import sys
import glob
import time
from tqdm import tqdm
import datetime  # 🔥 新增
import torch
import gc

# 引入已有的处理器类
from stage1_sam_segmentation import Stage1SAMProcessor
from stage2_clip_classification import Stage2CLIPProcessor

# ==============================================================================
# 配置区域
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 1. 原始街景图路径
STREETVIEW_DIR = os.path.join(PROJECT_ROOT, "data/input_streetview")

# 2. 眼动热力图根目录
HEATMAP_ROOT = os.path.join(PROJECT_ROOT, "data/experiment_data/gaze_heatmap")

# 3. CSV 目录
CSV_DIR = os.path.join(PROJECT_ROOT, "data/experiment_data/csv")

# 4. 输出目录
CACHE_ROOT = os.path.join(PROJECT_ROOT, "output/intermediate_cache")
FINAL_OUT_ROOT = os.path.join(PROJECT_ROOT, "output/final_results")
EXCEL_PATH = os.path.join(FINAL_OUT_ROOT, "Global_Analysis_Result_Batch.xlsx")

# 5. 模型路径
SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, "sam2_repo", "checkpoints", "sam2.1_hiera_large.pt")
CLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, "data/weights/clip")


# ==============================================================================
# 批处理逻辑
# ==============================================================================

def parse_heatmap_filename(filename):
    """
    解析热力图文件名，提取 Image ID。
    格式: 001_1_eyetrack_heatmap_...png -> Image ID = 1
    """
    try:
        parts = filename.split('_')
        img_id = parts[1]  # 获取中间的数字 ID
        return img_id
    except:
        return None


def scan_all_tasks():
    """扫描所有任务并返回任务列表"""
    print(f"🔍 Scanning tasks in: {HEATMAP_ROOT}")

    if not os.path.exists(HEATMAP_ROOT):
        print(f"❌ Error: Heatmap root path does not exist!")
        return []

    # 获取所有用户文件夹 (001, 002...)
    user_dirs = sorted([d for d in os.listdir(HEATMAP_ROOT) if os.path.isdir(os.path.join(HEATMAP_ROOT, d))])
    all_tasks = []

    print(f"   Found {len(user_dirs)} user directories.")

    for user_id in user_dirs:
        user_heatmap_dir = os.path.join(HEATMAP_ROOT, user_id)

        try:
            files = os.listdir(user_heatmap_dir)
        except Exception as e:
            print(f"   ⚠️ Cannot read dir {user_id}: {e}")
            continue

        # 筛选符合条件的文件
        heatmaps = [os.path.join(user_heatmap_dir, f) for f in files if
                    f.endswith('.png') and '_eyetrack_heatmap_' in f]

        for hm_path in heatmaps:
            hm_filename = os.path.basename(hm_path)
            img_id = parse_heatmap_filename(hm_filename)

            if not img_id:
                # print(f"      ⚠️ Filename parse failed: {hm_filename}")
                continue

            # 构造对应的街景图路径
            sv_name = f"QINCHUAN-{img_id}.jpg"
            sv_path = os.path.join(STREETVIEW_DIR, sv_name)

            # 兼容大小写 (.JPG)
            if not os.path.exists(sv_path):
                sv_path = os.path.join(STREETVIEW_DIR, f"QINCHUAN-{img_id}.JPG")

            # 再次检查 (用于调试)
            if not os.path.exists(sv_path):
                # print(f"      ⚠️ Streetview missing for ID {img_id} (User {user_id})")
                continue

            # 构造缓存路径
            sv_basename = os.path.splitext(os.path.basename(sv_path))[0]
            pkl_name = f"{sv_basename}_stage1.pkl"
            pkl_dir = os.path.join(CACHE_ROOT, user_id)
            pkl_path = os.path.join(pkl_dir, pkl_name)

            all_tasks.append({
                "user": user_id,
                "img_id": img_id,
                "hm_path": hm_path,
                "sv_path": sv_path,
                "pkl_path": pkl_path,
                "pkl_dir": pkl_dir
            })

    print(f"✅ Found {len(all_tasks)} valid tasks.")
    return all_tasks


def main1(all_tasks):
    # ---------------------------------------------------------
    # Phase 1: 批量运行 Stage 1 (SAM 分割)
    # ---------------------------------------------------------
    print("\n🚀 [Batch Stage 1] Checking SAM2 Segmentation Cache...")

    tasks_to_run_s1 = [t for t in all_tasks if not os.path.exists(t['pkl_path'])]

    if tasks_to_run_s1:
        print(f"   Note: Found {len(tasks_to_run_s1)} new tasks to segment. Loading SAM2...")

        if not os.path.exists(SAM_CHECKPOINT):
            print(f"❌ SAM Checkpoint not found: {SAM_CHECKPOINT}")
            return

        sam_processor = Stage1SAMProcessor(SAM_CHECKPOINT)

        for task in tqdm(tasks_to_run_s1, desc="Stage 1 Processing"):
            os.makedirs(task['pkl_dir'], exist_ok=True)
            try:
                sam_processor.run_segmentation(task['sv_path'], task['hm_path'], task['pkl_dir'])
            except Exception as e:
                print(f"❌ Stage 1 Error ({task['user']}-{task['img_id']}): {e}")

        del sam_processor
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Stage 1 All Done.")
    else:
        print("✅ All Stage 1 tasks are already cached. Skipping.")


def main2(all_tasks):
    # ---------------------------------------------------------
    # Phase 2: 批量运行 Stage 2 (CLIP 分类 & 汇总)
    # ---------------------------------------------------------
    print("\n🚀 [Batch Stage 2] Starting CLIP Classification & Analysis...")

    if not os.path.exists(CLIP_MODEL_PATH):
        print(f"❌ CLIP Model not found: {CLIP_MODEL_PATH}")
        return

    # 🔥 1. 创建统一的时间戳运行目录
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_dir = os.path.join(FINAL_OUT_ROOT, f"Final_Run_{run_timestamp}")
    os.makedirs(current_run_dir, exist_ok=True)

    # 🔥 2. Excel 保存路径设置在当前运行目录下
    run_excel_path = os.path.join(current_run_dir, "Global_Analysis_Result_Batch.xlsx")
    print(f"📂 Output Directory: {current_run_dir}")
    print(f"📊 Excel will be saved to: {run_excel_path}")

    # 🔥 3. 可视化开关配置 (在此处修改 True/False)
    viz_settings = {
        "viz_step0": False,  # 是否输出 Step 0 (簇与点)
        "viz_step1": False,  # 是否输出 Step 1 (单个物体详细图 - 数量巨大，慎开)
        "viz_step1_5": False,  # 是否输出 Step 1.5 (簇内共识分析)
        "viz_step2": True  # 是否输出 Step 2 (最终汇总图 - 建议开启)
    }

    clip_processor = Stage2CLIPProcessor(CLIP_MODEL_PATH)

    for task in tqdm(all_tasks, desc="Stage 2 Processing"):
        if not os.path.exists(task['pkl_path']):
            continue

        try:
            # 🔥 4. 为每个用户在 Run 目录下创建子文件夹 (例如: Final_Run_.../001/)
            user_final_out = os.path.join(current_run_dir, task['user'])
            os.makedirs(user_final_out, exist_ok=True)

            clip_processor.batch_run_classification_and_viz(
                pkl_path=task['pkl_path'],
                original_img_path=task['sv_path'],
                heatmap_path=task['hm_path'],
                output_dir=user_final_out,  # 传入具体的子文件夹
                global_excel_path=run_excel_path,  # 传入统一的 Excel 路径
                **viz_settings  # 传入可视化开关
            )

        except Exception as e:
            print(f"❌ Stage 2 Error ({task['user']}-{task['img_id']}): {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🎉 All Batch Processing Completed!")
    print(f"📂 Results saved in: {current_run_dir}")
    print(f"📊 Excel file: {run_excel_path}")


if __name__ == "__main__":
    # 1. 扫描一次任务列表
    tasks = scan_all_tasks()

    # =========================================================
    # 🔥 [修改] 断点续跑逻辑
    # 如果你想从 028 号志愿者开始跑，取消下面两行的注释
    # =========================================================
    start_user = "028"
    print(f"🔄 Resuming from user: {start_user} ...")
    tasks = [t for t in tasks if t['user'] >= start_user]
    print(f"✅ Tasks remaining: {len(tasks)}")
    # =========================================================

    if tasks:
        # 🔥 [控制开关] 根据需要注释/取消注释

        # 运行第一步：生成 SAM 分割缓存 (耗时, GPU 显存占用高)
        # main1(tasks)

        # 运行第二步：CLIP 分类 + 导出 Excel (较快, 需要 Stage 1 结果)
        main2(tasks)