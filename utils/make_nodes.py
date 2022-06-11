import onnx

def make_node(node_info):
    name = node_info['properties']['name']
    op_type = node_info['properties']['op_type']
    attributes = node_info['attributes']
    # attributes = {k: v for k, v in node_info['attributes'].items() if not v == 'undefined'}
    # print(attributes)
    
    inputs = []
    for key in node_info['inputs'].keys():
        inputs += node_info['inputs'][key]
    outputs = []
    for key in node_info['outputs'].keys():
        outputs += node_info['outputs'][key]
    
    # https://github.com/onnx/onnx/blob/main/onnx/helper.py#L82
    node = onnx.helper.make_node(
        op_type=op_type,
        inputs=inputs,
        outputs=outputs,
        name=name,
        **attributes
    )
    
    # print(node)
    
    return node