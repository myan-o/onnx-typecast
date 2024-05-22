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

def has_float16(data_type):
    return data_type == TensorProto.FLOAT or \
       data_type == TensorProto.DOUBLE or \
       data_type == TensorProto.UINT64 or \
       data_type == TensorProto.UINT32 or \
       data_type == TensorProto.UINT16 or \
       data_type == TensorProto.UINT8 or \
       data_type == TensorProto.INT64 or \
       data_type == TensorProto.INT32 or \
       data_type == TensorProto.INT16 or \
       data_type == TensorProto.INT8 \

def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def _convert_param_to_float16(param):
    data = param
    if has_float16(data.data_type):
        data_cvt = nph.to_array(data).astype(np.float16)
        data = nph.from_array(data_cvt, data.name)
    return data

def convert_params_to_float16(params_dict):
    with Pool() as pool:
        converted_params = pool.map(_convert_param_to_float16, params_dict.values())
    return converted_params

def _convert_constant_node_to_float16(node):
    # ノードが属性を持っていない、または属性が空の場合、そのままノードを返す
    if not hasattr(node, 'attribute') or len(node.attribute) == 0:
        return node

    # ノードの出力を変換
    new_inputs = []
    for inp in node.input:
        tensor = h.get_value_info(graph, inp).type.tensor_type
        if has_float16(tensor.data_type):
            tensor.data_type = onnx.TensorProto.FLOAT16
        new_inputs.append(inp)

    # ノードの出力を変換
    new_outputs = []
    for out in node.output:
        tensor = h.get_value_info(graph, out).type.tensor_type
        if has_float16(tensor.data_type):
            tensor.data_type = onnx.TensorProto.FLOAT16
        new_outputs.append(out)

    # 各属性を処理
    new_attributes = []
    for attr in node.attribute:
        new_attributes.append(attr)
        continue
        # TensorProto.FLOAT16 以外のデータ型のみを変換
        if hasattr(attr, 't') and has_float16(attr.t.data_type):
            data = nph.to_array(attr.t).astype(np.float16)
            new_t = nph.from_array(data)
            new_attr = h.make_attribute(attr.name, new_t)
            new_attributes.append(new_attr)
        else:
            new_attributes.append(attr)

    # 新しいノードを作成
    new_node = h.make_node(
        node.op_type,
        inputs=new_inputs,
        outputs=new_outputs,
        name=node.name,
        attributes=new_attributes,
    )
    return new_node


def convert_constant_nodes_to_float16(nodes):
    with Pool() as pool:
        new_nodes = pool.map(_convert_constant_node_to_float16, nodes)
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
