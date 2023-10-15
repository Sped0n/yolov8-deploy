import argparse
from pathlib import Path

import onnx
from onnx import shape_inference
from onnxsim import simplify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to model")
    parser.add_argument(
        "-o", "--output", default="./output.onnx", type=str, help="Path to output file"
    )
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # load model
    input_model = Path(args.input)
    try:
        assert input_model.is_file()
    except AssertionError:
        raise FileNotFoundError("Input file not found")
    model = onnx.load(str(input_model.absolute()))
    # load graph
    graph = model.graph

    # get input shape
    try:
        assert (
            graph.input[0].type.tensor_type.shape.dim[2].dim_value
            == graph.input[0].type.tensor_type.shape.dim[3].dim_value
        )
    except AssertionError:
        raise NotImplementedError("Dynamic shape is not supported")
    input_shape = graph.input[0].type.tensor_type.shape.dim[2].dim_value

    # use shape inference to get the number of classes
    inferred_model = shape_inference.infer_shapes(model)
    classes_num: int = (
        inferred_model.graph.output[0].type.tensor_type.shape.dim[1].dim_value
        - 4  # 4 is the output bounding box
    )

    print("==> Input shape: ", input_shape)
    print("==> Number of classes: ", classes_num)

    # remove the original output, usuallly outputs without
    # shape inference will be none, but we just remove it just in case
    graph.output.remove(graph.output[0])

    # we use tail flag to stop blind nodes removal after /model.22/Concat_3
    tail_flag = True

    for node_id, node in enumerate(reversed(graph.node)):
        if (
            node.name == "/model.22/Reshape"
            or node.name == "/model.22/Reshape_1"
            or node.name == "/model.22/Reshape_2"
            or tail_flag
        ):
            graph.node.remove(node)
        if node.name == "/model.22/Concat_3":
            tail_flag = False

    # after removing the blind nodes, we need to add the new outputs
    for idx in range(3):  # 3 outputs
        trp_insert_text = "" if idx == 0 else f"_{idx}"
        trp_tmp = onnx.helper.make_node(  # type: ignore
            name=f"/model.22/Transpose{trp_insert_text}",
            op_type="Transpose",
            inputs=[f"/model.22/Concat{trp_insert_text}_output_0"],
            outputs=[f"output{idx}"],  # output node name
            perm=[0, 2, 3, 1],
        )
        graph.node.append(trp_tmp)
        stride = 8 * (2**idx)
        out_tmp = onnx.helper.make_tensor_value_info(  # type: ignore
            name=f"output{idx}",
            elem_type=1,
            shape=[
                1,
                int(input_shape / stride),
                int(input_shape / stride),
                64 + classes_num,
            ],
        )
        graph.output.append(out_tmp)

    # check the modded model
    try:
        onnx.checker.check_model(model)  # type: ignore
    except onnx.checker.ValidationError:  # type: ignore
        raise RuntimeError("modded ONNX model could not be validated")
    print("==> modded ONNX model validated")

    # simplify the modded model
    model_simp, check = simplify(model)

    try:
        assert check, "Simplified ONNX model could not be validated"
    except AssertionError:
        raise AssertionError("onnx simplify failed")
    print("==> modded ONNX model simplified")

    if Path(args.output).is_file() and not args.force:
        raise RuntimeError(
            "output file already exists, use -f/--force flag to overwrite it"
        )
    onnx.save_model(model_simp, args.output)

    print(f"==> modded ONNX model saved to {Path(args.output).absolute()}")
