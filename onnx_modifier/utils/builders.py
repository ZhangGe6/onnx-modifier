import onnx
from onnx import AttributeProto
from .parse_tools import str2val, str2onnxdtype

def make_new_node(node_info):
    name = node_info['properties']['name']
    op_type = node_info['properties']['op_type']
    # attributes = node_info['attributes']
    # attributes = {k: v for k, v in node_info['attributes'].items() if not v == 'undefined'}
    attributes = {}
    for attr_name, attr_meta in node_info['attributes'].items():
        attr_value, attr_type = attr_meta
        if attr_value == 'undefined' or len(attr_value.replace(' ', '')) == 0:
            continue
        attributes[attr_name] = str2val(attr_value, attr_type)
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

def make_attr_changed_node(node, attr_change_info):
    # convert the changed attribute value into the type that is consistent with the original attribute
    # because AttributeProto is constructed barely based on the input value
    # https://github.com/onnx/onnx/blob/4e24b635c940801555bee574b4eb3a34cab9acd5/onnx/helper.py#L472
    def make_type_value(value, AttributeProto_type):
        attr_type = AttributeProto.AttributeType.Name(AttributeProto_type)
        if attr_type == "FLOAT":
            return float(value)
        elif attr_type == "INT":
            return int(value)
        elif attr_type == "STRING":
            return str(value)
        elif attr_type == "FLOATS":
            return str2val(value, "float[]")
        elif attr_type == "INTS":
            return str2val(value, "int[]")
        elif attr_type == "STRINGS":
            return str2val(value, "string[]")
        else:
            raise RuntimeError("type {} is not considered in current version. \
                               You can kindly report an issue for this problem. Thanks!".format(attr_type))

    new_attr = dict()
    for attr in node.attribute:
        # print(onnx.helper.get_attribute_value(attr))
        if attr.name in attr_change_info.keys():
            # attr_change_info: {attr: [value, type]}
            new_attr[attr.name] = make_type_value(attr_change_info[attr.name][0], attr.type)
        else:
            # https://github.com/onnx/onnx/blob/4e24b635c940801555bee574b4eb3a34cab9acd5/onnx/helper.py#L548
            new_attr[attr.name] = onnx.helper.get_attribute_value(attr)
    # print(new_attr)

    node = onnx.helper.make_node(
        op_type=node.op_type,
        inputs=node.input,
        outputs=node.output,
        name=node.name,
        **new_attr
    )

    # print(node)

    return node

def make_input(input_info: str):
    # ['input.4', 'float32[1,8,96,96]']
    name, type_ = input_info
    dtype = str2onnxdtype(type_.split("[")[0])
    shape = str2val(type_.split("[")[1].split("]")[0], "int[]")
    value_info = onnx.helper.make_tensor_value_info(
                                name, dtype, shape)

    return value_info