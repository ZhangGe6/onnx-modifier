import os
try:
    import onnx_tool
    print("module 'onnx-tool' is installed")
except ModuleNotFoundError:
    os.system("pip install onnx-tool==0.6.4")
    import onnx_tool

def shape_inference_using_onnx_tool(model_proto):
    g = onnx_tool.Graph(model_proto.graph, verbose=False)
    g.shape_infer()
    
    value_protos = []
    for key in g.dynamics:
        tensor = g.tensormap[key]
        vinfo = tensor.make_value_proto()
        if vinfo is None:
            continue
        if vinfo not in value_protos:
            value_protos.append(vinfo)

    return value_protos