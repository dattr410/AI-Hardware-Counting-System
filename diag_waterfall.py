# -*- coding: utf-8 -*-
"""
Chan doan offline: vat bi mat o TANG NAO (model NMS / NMS-free / dedup / ghep frame)?

Cach dung (moi phien = 1 thu muc co frame_1.png ... frame_5.png, tao boi COUNT_DEBUG=1):
  python diag_waterfall.py static/saved_images/debug/session_X --gt-screw 7
  python diag_waterfall.py static/saved_images/debug/session_X --gt-screw 7 --profiles baseline,v2,v3
  python diag_waterfall.py "static/saved_images/debug/*" --profiles baseline,v3   (nhieu phien, khong GT)

Doc ket qua:
  nms_free > model_nms           -> NMS cua model dang suppress nham
  after_dedup < truoc dedup      -> dedup dang gop nham
  nms_free@0.10 >= GT nhung @0.30 < GT -> vat co that, conf thap: can xac nhan qua frame
  ca 5 frame @0.10 van < GT      -> detector khong thay: kiem tra imgsz/tien xu ly/du lieu train
  du so box nhung 1 box phu 2 vit -> gioi han box ngang -> can OBB/seg
"""
import argparse
import glob
import os

import cv2

import app as A


def load_session(folder):
    paths = sorted(glob.glob(os.path.join(folder, "frame_*.png")))
    return [cv2.imread(p) for p in paths if cv2.imread(p) is not None]


def n_cls(dets, name="screw"):
    return sum(1 for d in dets if A.cname(d["class_id"]) == name)


def waterfall(frames, gt, cls="screw"):
    if not A.E2E_KW:
        print("[WARN] Khong nhan dien duoc tham so NMS-free; cot nms_free = mac dinh cua model.")
    print(f"  ultralytics e2e kwargs = {A.E2E_KW}")
    print(f"  {'frame':<6}{'GT':>4}{'model_nms':>11}{'nms_free':>10}{'nms+dedup':>11}{'free+dedup':>12}"
          f"{'free+geo':>10}{'free@0.10':>11}")
    for i, fr in enumerate(frames, 1):
        h, w = fr.shape[:2]
        x1, y1, x2, y2 = A.get_roi_px(w, h)
        crop = fr[y1:y2, x1:x2]
        src = A.preprocess_frame(crop) if A.USE_PREPROCESS else crop

        def run(**kw):
            r = A.model.predict(src, imgsz=A.IMAGE_SIZE, max_det=300, verbose=False, **kw)[0]
            dets = A.get_detections(r)
            for d in dets:
                b = d["box"]
                d["box"] = [b[0] + x1, b[1] + y1, b[2] + x1, b[3] + y1]
            return dets

        a = run(conf=0.30, iou=0.5)
        b = run(conf=0.30, **A.E2E_KW)
        c = A.deduplicate(a, 0.65)
        d = A.deduplicate(b, 0.65)
        e = A.geo_dedup(b, fr)
        f = run(conf=0.10, **A.E2E_KW)
        print(f"  {i:<6}{gt if gt is not None else '-':>4}{n_cls(a, cls):>11}{n_cls(b, cls):>10}"
              f"{n_cls(c, cls):>11}{n_cls(d, cls):>12}{n_cls(e, cls):>10}{n_cls(f, cls):>11}")


def compare_profiles(frames, profiles):
    print(f"  {'profile':<10}{'nut':>5}{'screw':>7}{'washer':>8}{'total':>7}{'pending':>9}{'time_s':>8}  review")
    for p in profiles:
        res = A.run_counting_pipeline(frames, A.make_cfg(p))
        c = res["counts"]
        print(f"  {p:<10}{c.get('nut', 0):>5}{c.get('screw', 0):>7}{c.get('washer', 0):>8}"
              f"{res['total']:>7}{res['pending']:>9}{res['t_total']:>8.2f}  {len(res['reasons'])} flag(s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", help="thu muc phien (ho tro glob)")
    ap.add_argument("--gt-screw", type=int, default=None, help="so vit that (neu co)")
    ap.add_argument("--profiles", default="", help="vd baseline,v2,v3")
    args = ap.parse_args()

    for folder in sorted(glob.glob(args.sessions)):
        frames = load_session(folder)
        if not frames:
            continue
        print(f"\n=== {folder} ({len(frames)} frame) ===")
        if args.gt_screw is not None or not args.profiles:
            waterfall(frames, args.gt_screw)
        if args.profiles:
            compare_profiles(frames, [p.strip() for p in args.profiles.split(",") if p.strip()])


if __name__ == "__main__":
    main()
