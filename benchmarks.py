import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  

import export
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from segment.val import run as val_seg
from utils import notebook_init
from utils.general import LOGGER, check_yaml, file_size, print_args
from utils.torch_utils import select_device
from val import run as val_det


def run(
    weights=ROOT / "yolov5s.pt",  
    imgsz=640,  
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml",  
    device="",  
    half=False, 
    test=False,  
    pt_only=False,  
    hard_fail=False,  
):
    y, t = [], time.time()
    device = select_device(device)
    model_type = type(attempt_load(weights, fuse=False))  
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  
        try:
            assert i not in (9, 10), "inference not supported"  
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            # Export
            if f == "-":
                w = weights  
            else:
                w = export.run(
                    weights=weights, imgsz=[imgsz], include=[f], batch_size=batch_size, device=device, half=half
                )[-1]  
            assert suffix in str(w), "export failed"

            # Validate
            if model_type == SegmentationModel:
                result = val_seg(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][7] 
            else:  # DetectionModel:
                result = val_det(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][3]  
            speed = result[2][1]  
            y.append([name, round(file_size(w), 1), round(metric, 4), round(speed, 2)])  
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f"Benchmark --hard-fail for {name}: {e}"
            LOGGER.warning(f"WARNING ⚠️ Benchmark failure for {name}: {e}")
            y.append([name, None, None, None])  # mAP, t_inference
        if pt_only and i == 0:
            break  

    # Print results
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  
    c = ["Format", "Size (MB)", "mAP50-95", "Inference time (ms)"] if map else ["Format", "Export", "", ""]
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f"\nBenchmarks complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py if map else py.iloc[:, :2]))
    if hard_fail and isinstance(hard_fail, str):
        metrics = py["mAP50-95"].array  
        floor = eval(hard_fail) 
        assert all(x > floor for x in metrics if pd.notna(x)), f"HARD FAIL: mAP50-95 < floor {floor}"
    return py


def test(
    weights=ROOT / "yolov5s.pt", 
    imgsz=640,  # inference size (pixels)
    batch_size=1,  # batch size
    data=ROOT / "data/coco128.yaml", 
    device="",  
    half=False, 
    test=False, 
    pt_only=False, 
    hard_fail=False,  
):
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():  
        try:
            w = (
                weights
                if f == "-"
                else export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]
            )  # weights
            assert suffix in str(w), "export failed"
            y.append([name, True])
        except Exception:
            y.append([name, False])  

    # Print results
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  
    py = pd.DataFrame(y, columns=["Format", "Export"])
    LOGGER.info(f"\nExports complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py))
    return py


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--test", action="store_true", help="test exports only")
    parser.add_argument("--pt-only", action="store_true", help="test PyTorch only")
    parser.add_argument("--hard-fail", nargs="?", const=True, default=False, help="Exception on error or < min metric")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
