import onnx

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto, GraphProto
from onnx import numpy_helper as nph

import numpy as np
from collections import OrderedDict

from logger import log
import typer


def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def convert_params_to_float16(params_dict):
    converted_params = []
    for param in params_dict:
        data = params_dict[param]
        if data.data_type != TensorProto.FLOAT16:
            data_cvt = nph.to_array(data).astype(np.float16)
            data = nph.from_array(data_cvt, data.name)
        converted_params += [data]
    return converted_params

def convert_constant_nodes_to_float16(nodes):
    """
    convert_constant_nodes_to_float16 Convert Constant nodes to INT32. If a constant node has data type INT64, a new version of the
    node is created with INT32 data type and stored.

    Args:
        nodes (list): list of nodes

    Returns:
        list: list of new nodes all with INT32 constants.
    """
    new_nodes = []
    for node in nodes:
        if node.attribute[0].t.data_type != TensorProto.FLOAT16:
            data = nph.to_array(node.attribute[0].t).astype(np.float16)
            new_t = nph.from_array(data)
            new_node = h.make_node(
                node.op_type,
                inputs=[],
                outputs=node.output,
                name=node.name,
                value=new_t,
            )
            new_nodes += [new_node]
        else:
            new_nodes += [node]

    return new_nodes


def convert_model_to_float16(model_path: str, out_path: str):
    """
    convert_model_to_float16 Converts ONNX model with INT64 params to INT32 params.\n

    Args:\n
        model_path (str): path to original ONNX model.\n
        out_path (str): path to save converted model.
    """
    log.info("ONNX FLOAT16 Converter")
    log.info(f"Loading Model: {model_path}")
    # * load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    # * get model opset version.
    opset_version = model.opset_import[0].version
    graph = model.graph
    # * The initializer holds all non-constant weights.
    init = graph.initializer
    # * collect model params in a dictionary.
    params_dict = make_param_dictionary(init)
    log.info("Converting model params to FLOAT16...")
    # * convert all params to FLOAT16.
    converted_params = convert_params_to_float16(params_dict)
    log.info("Converting nodes to FLOAT16...")
    new_nodes = convert_constant_nodes_to_float16(graph.node)

    graph_name = f"{graph.name}-float16"
    log.info("Creating new graph...")
    # * create a new graph with converted params and new nodes.
    graph_float16 = h.make_graph(
        new_nodes,
        graph_name,
        graph.input,
        graph.output,
        initializer=converted_params,
    )
    log.info("Creating new float16 model...")
    model_float16 = h.make_model(graph_float16, producer_name="onnx-typecast")
    model_float16.opset_import[0].version = opset_version
    ch.check_model(model_float16)
    log.info(f"Saving converted model as: {out_path}")
    onnx.save_model(model_float16, out_path)
    log.info(f"Done Done London. 🎉")
    return


if __name__ == "__main__":
    typer.run(convert_model_to_float16)
