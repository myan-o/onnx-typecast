import onnx

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto, GraphProto
from onnx import numpy_helper as nph

import numpy as np
from collections import OrderedDict
from multiprocessing import Pool

from logger import log
import typer

new_inputs = []
new_outputs = []

def find_by_name(_list, name):
    for item in _list:
        if item.name == name:
            return item
    return None


def can_convert_float16(data_type):
    return data_type == TensorProto.FLOAT or \
       data_type == TensorProto.DOUBLE or \
       data_type == TensorProto.UINT64 or \
       data_type == TensorProto.UINT32 or \
       data_type == TensorProto.UINT16 or \
       data_type == TensorProto.UINT8 or \
       data_type == TensorProto.INT64 or \
       data_type == TensorProto.INT32 or \
       data_type == TensorProto.INT16 or \
       data_type == TensorProto.INT8

def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def _convert_param_to_float16(param):
    data = param
    if can_convert_float16(data.data_type):
        data_cvt = nph.to_array(data).astype(np.float16)
        data = nph.from_array(data_cvt, data.name)
    return data

def convert_params_to_float16(params_dict):
    with Pool() as pool:
        converted_params = pool.map(_convert_param_to_float16, params_dict.values())
    return converted_params

def _convert_constant_nodes_to_float16(node):
    global graph
    global new_inputs
    global new_outputs

    # ノードの入力をキャスト
    for name in node.input:
        if find_by_name(new_inputs, name) is not None:
            continue

        vi = find_by_name(graph.input, name)
        if vi is None:
            continue

        if can_convert_float16(vi.type.tensor_type.elem_type):
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            new_inputs.append(h.make_tensor_value_info(vi.name, onnx.TensorProto.FLOAT16, shape))
        else:
            new_inputs.append(vi)

    # ノードの出力をキャスト
    for name in node.output:
        if find_by_name(new_outputs, name) is not None:
            continue

        vi = find_by_name(graph.output, name)
        if vi is None:
            continue

        if can_convert_float16(vi.type.tensor_type.elem_type):
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            new_outputs.append(h.make_tensor_value_info(vi.name, onnx.TensorProto.FLOAT16, shape))
        else:
            new_outputs.append(vi)

    new_node = h.make_node(
        op_type = node.op_type,
        inputs  = node.input,
        outputs = node.output,
        name    = node.name,
    )

    # ノードの属性(定数)をキャスト
    if hasattr(node, 'attribute'):
        for attr in node.attribute:
            if hasattr(attr, 't') and can_convert_float16(attr.t.data_type):
                data = nph.to_array(attr.t).astype(np.float16)
                new_t = nph.from_array(data)
                new_attr = h.make_attribute(attr.name, new_t)
                new_node.attribute.append(new_attr)
            else:
                new_node.attribute.append(attr)

    return new_node

def convert_constant_nodes_to_float16(nodes):
#    with Pool() as pool:
#        new_nodes = pool.map(_convert_constant_nodes_to_float16, nodes)
    new_nodes = []
    for node in nodes:
        new_nodes.append(_convert_constant_nodes_to_float16(node))
    return new_nodes

def convert_model_to_float16(model_path: str, out_path: str):
    global graph
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
        new_inputs,
        new_outputs,
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
