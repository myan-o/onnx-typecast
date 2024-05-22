import onnx

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto, GraphProto
from onnx import numpy_helper as nph

import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import log
import typer


def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def convert_param_to_float16(param):
    data = param
    if data.data_type != TensorProto.FLOAT16:
        data_cvt = nph.to_array(data).astype(np.float16)
        data = nph.from_array(data_cvt, data.name)
    return data


def convert_params_to_float16(params_dict):
    converted_params = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(convert_param_to_float16, params_dict[param]): param for param in params_dict}
        for future in as_completed(futures):
            converted_params.append(future.result())
    return converted_params


def convert_constant_node_to_float16(node):
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
        return new_node
    return node


def convert_constant_nodes_to_float16(nodes):
    new_nodes = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(convert_constant_node_to_float16, node): node for node in nodes}
        for future in as_completed(futures):
            new_nodes.append(future.result())
    return new_nodes


def convert_model_to_float16(model_path: str, out_path: str):
    log.info("ONNX FLOAT16 Converter")
    log.info(f"Loading Model: {model_path}")
    model = onnx.load_model(model_path)
    ch.check_model(model)
    opset_version = model.opset_import[0].version
    graph = model.graph
    init = graph.initializer
    params_dict = make_param_dictionary(init)
    log.info("Converting model params to FLOAT16...")
    converted_params = convert_params_to_float16(params_dict)
    log.info("Converting nodes to FLOAT16...")
    new_nodes = convert_constant_nodes_to_float16(graph.node)

    graph_name = f"{graph.name}-float16"
    log.info("Creating new graph...")
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
