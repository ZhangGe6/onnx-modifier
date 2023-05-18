import os

def add_outputs_using_onnx_tool(output_names, model_proto):
    try:
        # pip page: https://pypi.org/project/onnx-tool/
        # github page: https://github.com/ThanatosShinji/onnx-tool 
        import onnx_tool
        print("module 'onnx-tool' is installed")
    except ModuleNotFoundError:
        os.system("pip install onnx-tool==0.6.4")
        import onnx_tool
    
    g = onnx_tool.Graph(model_proto.graph, verbose=False)
    g.shape_infer()
    
    out_protos = []
    for name in output_names:
        if name in g.tensormap:
            proto = g.tensormap[name].make_value_proto()
            out_protos.append(proto)
    
    return out_protos