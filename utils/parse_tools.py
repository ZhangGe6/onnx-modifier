import numpy as np
from typing import cast
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx import TensorProto

# parse numpy values from string
def parse_str2np(tensor_str, tensor_type):
    def parse_value(value_str, value_type):
        if value_type.startswith('int'):
            return int(value_str)
        elif value_type.startswith('float'):
            return float(value_str)
        else:
            raise RuntimeError("type {} is not considered in current version. \
                                You can kindly report an issue for this problem. Thanks!".format(value_type))

    def extract_val():
        num_str = ""
        while (len(stk) > 0) and (type(stk[-1]) == str and ord('0') <= ord(stk[-1]) <= ord('9') or stk[-1] in ['+', '-', '.', 'e', 'E']):
            num_str = stk.pop() + num_str
        
        if len(num_str) > 0:
            return parse_value(num_str, tensor_type)
        else:
            return None
    
    tensor_str = tensor_str.replace(" ", "")
    tensor_str = tensor_str.replace("\n", "")
    stk = []
    for c in tensor_str: # '['  ','  ']' '.' '-' or value
        if c == ",":
            ext_val = extract_val()
            if ext_val is not None: stk.append(ext_val)
        elif c == "]":
            ext_val = extract_val()
            if ext_val is not None: stk.append(ext_val)
            
            arr = []
            while stk[-1] != '[':
                arr.append(stk.pop())
            stk.pop()  # the left [
            
            arr.reverse()
            stk.append(arr)
        else:
            stk.append(c)
    val = stk[0]
    
    # wrap with numpy with the specific data type
    if tensor_type == "int64":
        return np.array(val, dtype=np.int64)
    elif tensor_type == "int32":
        return np.array(val, dtype=np.int32)
    elif tensor_type == "int8":
        return np.array(val, dtype=np.int8)
    elif tensor_type == "float64":
        return np.array(val, dtype=np.float64)
    elif tensor_type == "float32":
        return np.array(val, dtype=np.float32)
    else:
        raise RuntimeError("type {} is not considered in current version. \
                            You can kindly report an issue for this problem. Thanks!".format(tensor_type))
    
# parse Python or onnx built-in values from string
def parse_str2val(val_str, val_type):
    # a compatible function in case user inputs double side bracket for list values
    def rm_doubleside_brackets(ls_val_str):
        return ls_val_str[1:-1]
        
    # Python built-in values
    if val_type in ["int", "int32", "int64"]:
        return int(val_str)
    elif val_type in ["int[]", "int32[]", "int64[]"]:
        attr_val = []
        for v in rm_doubleside_brackets(val_str).split(","):
            attr_val.append(int(v))
        return attr_val
    elif val_type in ["float", "float32", "float64"]:
        return float(val_str)
    elif val_type in ["float[]", "float32[]", "float64[]"]:
        attr_val = []
        for v in rm_doubleside_brackets(val_str).split(","):
            attr_val.append(float(v))
        return attr_val
    
    # onnx built-in values 
    elif val_type == "DataType":
        # https://github.com/onnx/onnx/blob/46b96275554b0d978dd5c8ba786cc81dabd1d25a/onnx/onnx.proto#L479
        return getattr(TensorProto, val_str.upper())
        
    else:
        raise RuntimeError("type {} is not considered in current version. \
                            You can kindly report an issue for this problem. Thanks!".format(val_type))


# map np datatype to onnx datatype
# https://github.com/onnx/onnx/blob/8669fad0247799f4d8683550eec749974b4f5338/onnx/helper.py#L1177
def np2onnxdtype(np_dtype):
    return cast(int, NP_TYPE_TO_TENSOR_TYPE[np_dtype])

if __name__ == "__main__":
    def test_parse_str2np():
        # tensor_str = "1"
        # tensor_str = "[1, 2, 3]"
        tensor_str = "[[10, 2.3, 3],[1, 2e6, 3]]"
        val = parse_str2np(tensor_str, "float32")
        print(type(val), val)
        
        tensor_str = "[[10, 2, 3],[1, 2, 3]]"
        val = parse_str2np(tensor_str, "int64")
        print(type(val), val)
    # test_parse_str2np()
    
    def test_parse_str2val():
        val_str = "1"
        val = parse_str2val(val_str, "int")
        print(val)
        
        # val_str = "1, 2, 3"
        val_str = "[1, 2, 3]"
        val = parse_str2val(val_str, "int[]")
        print(val)
        val = parse_str2val(val_str, "float[]")
        print(val)
        
        val_str = "int8"
        # val_str = "float"
        val = parse_str2val(val_str, "DataType")
        print(val)
    test_parse_str2val()
    
    