import copy
import logging
import onnx
import onnx_tool

# logging.basicConfig(level=logging.INFO)

def shape_inference_using_onnx_tool(model_proto):
    g = onnx_tool.Graph(model_proto.graph, verbose=False)
    g.shape_infer()

    inferred_value_info = []
    for key in g.dynamics:
        tensor = g.tensormap[key]
        vinfo = tensor.make_value_proto()
        if vinfo is None:
            continue
        if vinfo not in inferred_value_info:
            inferred_value_info.append(vinfo)

    return inferred_value_info

def shape_inference_primitive(model_proto):
    shape_info = onnx.shape_inference.infer_shapes(model_proto)
    inferred_value_info = [v for v in shape_info.graph.value_info]

    return inferred_value_info

def get_infered_shape(model_proto):
    inferred_value_info = None
    logging.warning("[EXPERIMENTAL] Do shape inference automatically...")
    reset_model_proto = copy.deepcopy(model_proto)
    value_info_bak = copy.deepcopy(reset_model_proto.graph.value_info)
    del reset_model_proto.graph.value_info[:]
    # del reset_model_proto.graph.output[:]
    try:
        inferred_value_info = shape_inference_using_onnx_tool(reset_model_proto)
    except:
        logging.warning("shape inference using onnx-tool fails, fallback to primitive ONNX Python API.")
        # avoid empty value_info for onnx.shape_inference.infer_shapes
        reset_model_proto.graph.value_info.extend(value_info_bak)
        inferred_value_info = shape_inference_primitive(reset_model_proto)

    return inferred_value_info