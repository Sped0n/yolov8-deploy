import argparse
import dataclasses
import math
import random
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime


@dataclasses.dataclass
class PreprocessParams:
    w0: int
    h0: int
    w1: int
    h1: int
    wpad: int
    hpad: int
    scale: float
    dw: int
    dh: int


@dataclasses.dataclass
class RawDiagBbox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclasses.dataclass
class DiagBbox:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclasses.dataclass
class RawObjDetected:
    box: RawDiagBbox
    score: float
    clsid: int


@dataclasses.dataclass
class ObjDetected:
    box: DiagBbox
    score: float
    clsid: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Path to model")
    parser.add_argument("-i", "--input", type=str, help="Path to input image")
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    sigmoid function
    :param x: input array
    :return: sigmoided array
    """
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    softmax function
    :param x: input array
    :param axis: axis to apply softmax
    :return: softmaxed array
    """
    e_x: np.ndarray = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y: np.ndarray = e_x / e_x.sum(axis=axis, keepdims=True)
    return y


def preprocess_params_gen(frame: np.ndarray, input_size: int) -> PreprocessParams:
    """
    generate preprocess params, like params for resizing and padding
    :param frame: input frame
    :param input_size: input size
    :return: preprocess params
    """
    params = PreprocessParams(
        w0=frame.shape[1],
        h0=frame.shape[0],
        w1=-1,
        h1=-1,
        wpad=-1,
        hpad=-1,
        scale=-1.0,
        dw=-1,
        dh=-1,
    )

    if params.w0 > params.h0:
        params.scale = float(input_size / params.w0)
        params.w1 = input_size
        params.h1 = int(params.h0 * params.scale)
        params.wpad = 0
        params.hpad = input_size - params.h1
    else:
        params.scale = float(input_size / params.h0)
        params.h1 = input_size
        params.w1 = int(params.w0 * params.scale)
        params.hpad = 0
        params.wpad = input_size - params.w1
    params.dw = params.wpad // 2
    params.dh = params.hpad // 2
    return params


def pre_process(
    img: np.ndarray, input_shape: tuple[int, int, int, int]
) -> tuple[np.ndarray, PreprocessParams]:
    """
    preprocess input image
    :param image: input image
    :param input_size: input size
    :return: preprocessed image, preprocess params
    """
    pps_params: PreprocessParams = preprocess_params_gen(img, input_shape[2])
    # algorithm for resizing image
    interp = cv2.INTER_LINEAR if (pps_params.scale > 1) else cv2.INTER_AREA
    # resize while keeping aspect ratio
    resized: np.ndarray = cv2.resize(
        img, (pps_params.w1, pps_params.h1), interpolation=interp
    )
    # convert to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # pad image
    padded: np.ndarray = cv2.copyMakeBorder(
        resized,
        pps_params.dh,
        pps_params.hpad - pps_params.dh,
        pps_params.dw,
        pps_params.wpad - pps_params.dw,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    # convert to tensor
    tensor: np.ndarray = np.array(padded) / 255.0  # type: ignore
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)
    return tensor, pps_params


def box_translator(bbox_in: RawDiagBbox, params: PreprocessParams) -> DiagBbox:
    """
    translate bbox from resized image to original image
    :param bbox_in: bbox in resized image
    :param params: preprocess params
    :return: bbox in original image
    """
    # get bbox
    x0, y0, x1, y1 = bbox_in.x0, bbox_in.y0, bbox_in.x1, bbox_in.y1

    # scale back
    x0 = (x0 - params.dw) / params.scale
    y0 = (y0 - params.dh) / params.scale
    x1 = (x1 - params.dw) / params.scale
    y1 = (y1 - params.dh) / params.scale

    # clip image
    x0 = min(max(x0, 1), params.w0 - 1)
    y0 = min(max(y0, 1), params.h0 - 1)
    x1 = min(max(x1, 1), params.w0 - 1)
    y1 = min(max(y1, 1), params.h0 - 1)

    return DiagBbox(math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))


def post_process(
    mats: list[np.ndarray],
    pps_params: PreprocessParams,
    conf_thres: float = 0.25,
    nms_thres: float = 0.65,
    reg_max: int = 16,
) -> list[ObjDetected]:
    """
    postprocess output of model
    :param mats: output of model
    :param pps_params: preprocess params
    :param conf_thres: confidence threshold
    :param nms_thres: nms threshold
    :param reg_max: regression max
    """
    # dfl
    dfl: np.ndarray = np.arange(0, reg_max, dtype=np.float32)
    # raw scores, boxes and labels
    raw_scores: list[float] = []
    raw_boxes: list[np.ndarray] = []
    raw_labels: list[int] = []
    # iterate over output
    for i, mat in enumerate(mats):
        if mat.ndim == 3:
            pass
        elif mat.ndim == 4 and mat.shape[0] == 1:
            mat = mat[0]
        else:
            exit(0)
        stride = 8 << i
        boxes_feat, clses_feat = np.split(
            mat,
            [
                64,
            ],
            -1,
        )
        clses_feat = sigmoid(clses_feat)
        _argmax = clses_feat.argmax(-1)
        _max = clses_feat.max(-1)

        indices = np.where(_max > conf_thres)
        hi, wi = indices
        num_proposal = hi.size
        if not num_proposal:
            continue

        clses = _max[hi, wi]
        boxes = boxes_feat[hi, wi].reshape(-1, 4, reg_max)
        boxes = softmax(boxes, -1) @ dfl
        argmax = _argmax[hi, wi]

        for j in range(num_proposal):
            h, w = hi[j], wi[j]
            cls = clses[j]

            x0, y0, x1, y1 = boxes[j]

            x0 = (w + 0.5 - x0) * stride
            y0 = (h + 0.5 - y0) * stride
            x1 = (w + 0.5 + x1) * stride
            y1 = (h + 0.5 + y1) * stride

            clsid = argmax[j]

            raw_scores.append(float(cls))
            raw_boxes.append(np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32))
            raw_labels.append(clsid)
    # non maximum suppression
    nms_indices: Sequence[int] = cv2.dnn.NMSBoxesBatched(
        raw_boxes, raw_scores, raw_labels, conf_thres, nms_thres  # type: ignore
    )
    results: list[ObjDetected] = []
    for idx in nms_indices:
        tmp = raw_boxes[idx]
        # xywh to xyxy
        tmp[2:] = tmp[:2] + tmp[2:]
        results.append(
            ObjDetected(
                box=box_translator(
                    RawDiagBbox(tmp[0], tmp[1], tmp[2], tmp[3]), pps_params
                ),
                score=raw_scores[idx],
                clsid=raw_labels[idx],
            )
        )
    return results


if __name__ == "__main__":
    args = parse_args()
    # load model path
    model_path = Path(args.model)
    try:
        assert model_path.is_file()
    except AssertionError:
        raise FileNotFoundError("Input model not found")
    # load input image path
    image_path = Path(args.input)
    try:
        assert model_path.is_file()
    except AssertionError:
        raise FileNotFoundError("Input image not found")

    # load model
    session = onnxruntime.InferenceSession(str(model_path.absolute()))
    input_shape: tuple[int, int, int, int] = session.get_inputs()[0].shape
    print("==> input shape: ", input_shape)
    try:
        assert input_shape[2] == input_shape[3]
    except AssertionError:
        raise NotImplementedError("Dynamic shape is not supported")

    # load image
    image = cv2.imread(str(image_path.absolute()))

    # preprocess
    t0 = time.time()
    tensor, pps_params = pre_process(image, input_shape)

    # inference
    t1 = time.time()
    mat1, mat2, mat3 = session.run(
        ["output0", "output1", "output2"], {"images": tensor}
    )
    t2 = time.time()
    outputs = [np.array(mat1), np.array(mat2), np.array(mat3)]

    # postprocess
    results = post_process(outputs, pps_params)
    t3 = time.time()

    # display
    colors_80: list[tuple[int, int, int]] = [  # type: ignore
        tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(80)
    ]
    for result in results:
        color = colors_80[result.clsid]
        cv2.rectangle(
            image,
            (result.box.x0, result.box.y0),
            (result.box.x1, result.box.y1),
            color,
            2,
        )
        cv2.putText(
            image,
            f"{result.clsid}: {result.score:.2f}({(result.box.x0 + result.box.x1) / 2 - 640},{(result.box.y0 + result.box.y1) / 2 - 360}",  # noqa: E501
            (result.box.x0, result.box.y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    cv2.imshow("results", image)

    # print
    print(f"==> preprocess time: {(t1 - t0) * 1000:.2f}ms")
    print(f"==> inference time: {(t2 - t1) * 1000:.2f}ms")
    print(f"==> postprocess time: {(t3 - t2) * 1000:.2f}ms")
    # cv2 window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
