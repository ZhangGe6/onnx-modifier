import onnx

def make_node(node_info):
    name = node_info['properties']['name']
    op_type = node_info['properties']['op_type']
    # attributes = node_info['attributes']
    attributes = {k: v for k, v in node_info['attributes'].items() if not v == 'undefined'}
    # print(attributes)
    
    inputs = []
    for key in node_info['inputs'].keys():
        for inp in node_info['inputs'][key]:
            # filter out the un-filled io in list 
            if not inp.startswith('list_custom'):
                inputs.append(inp)
    outputs = []
    for key in node_info['outputs'].keys():
        for out in node_info['outputs'][key]:
            # filter out the un-filled io in list 
            if not out.startswith('list_custom'):
                outputs.append(out)
    
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