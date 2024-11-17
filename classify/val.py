
import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  

from models.common import DetectMultiBackend
from utils.dataloaders import create_classification_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    Profile,
    check_img_size,
    check_requirements,
    colorstr,
    increment_path,
    print_args,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    data=ROOT / "../datasets/mnist",  
    weights=ROOT / "yolov5s-cls.pt",  
    batch_size=128,  
    imgsz=224,  
    device="",  
    workers=8,  
    verbose=False,  
    project=ROOT / "runs/val-cls",  
    name="exp",  
    exist_ok=False,  
    half=False,  
    dnn=False,  
    model=None,
    dataloader=None,
    criterion=None,
    pbar=None,
):
    
    # Initialize/load model and set device
    training = model is not None
    if training:  
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  
        half &= device.type != "cpu"  
        model.half() if half else model.float()
    else:  
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  
        save_dir.mkdir(parents=True, exist_ok=True) 

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  
        half = model.fp16  
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  
                LOGGER.info(f"Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        # Dataloader
        data = Path(data)
        test_dir = data / "test" if (data / "test").exists() else data / "val"  
        dataloader = create_classification_dataloader(
            path=test_dir, imgsz=imgsz, batch_size=batch_size, augment=False, rank=-1, workers=workers
        )

    model.eval()
    pred, targets, loss, dt = [], [], 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    n = len(dataloader)  
    action = "validating" if dataloader.dataset.root.stem == "val" else "testing"
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)
    with torch.cuda.amp.autocast(enabled=device.type != "cpu"):
        for images, labels in bar:
            with dt[0]:
                images, labels = images.to(device, non_blocking=True), labels.to(device)

            with dt[1]:
                y = model(images)

            with dt[2]:
                pred.append(y.argsort(1, descending=True)[:, :5])
                targets.append(labels)
                if criterion:
                    loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    if verbose:  
        LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
        LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
        for i, c in model.names.items():
            acc_i = acc[targets == i]
            top1i, top5i = acc_i.mean(0).tolist()
            LOGGER.info(f"{c:>24}{acc_i.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")

        # Print results
        t = tuple(x.t / len(dataloader.dataset.samples) * 1e3 for x in dt)  
        shape = (1, 3, imgsz, imgsz)
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}" % t)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    return top1, top5, loss


def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "../datasets/mnist", help="dataset path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s-cls.pt", help="model.pt path(s)")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="inference size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--verbose", nargs="?", const=True, default=True, help="verbose output")
    parser.add_argument("--project", default=ROOT / "runs/val-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
