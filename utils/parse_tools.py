import numpy as np
from typing import cast
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx import TensorProto

# parse numpy values from string
def parse_str2np(tensor_str, tensor_type):
    def parse_value(value_str, value_type):
        if value_type.startswith('int') or value_type.startswith('uint'):
            return int(value_str)
        elif value_type.startswith('float'):
            return float(value_str)
        else:
            raise RuntimeError("type {} is not considered in current version.\n \
                                You are kindly to report an issue for this problem. Thanks!".format(value_type))

    def extract_val():
        num_str = ""
        while (len(stk) > 0) and (type(stk[-1]) == str and ord('0') <= ord(stk[-1]) <= ord('9') or stk[-1] in ['+', '-', '.', 'e', 'E']):
            num_str = stk.pop() + num_str
        
        if len(num_str) > 0:
            return parse_value(num_str, tensor_type)
        else:
            return None
    
    # preprocess for tensor_str: remove blank and newline character
    tensor_str = tensor_str.replace(" ", "")
    tensor_str = tensor_str.replace("\n", "")
    # preprocess for tensor_type: extract type info in case users input type+shape, like `float32[1,3,1,1]``
    tensor_type = tensor_type.split("[")[0]
    # for vector
    if "[" in tensor_str:
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
    # for scalar
    else:
        val = tensor_str
    
    # wrap with numpy with the specific data type
    try:
        return np.array(val, getattr(np, tensor_type))
    except:
        raise RuntimeError("Type {} is not supported.\n \
                            You can check all supported datatypes in \
                                https://numpy.org/doc/stable/user/basics.types.html or \
                                https://www.tutorialspoint.com/numpy/numpy_data_types.htm . \
                            If the problem still exists, \
                            you are kindly to report an issue. Thanks!".format(tensor_type))
    
# parse Python or onnx built-in values from string
def parse_str2val(val_str, val_type):
    def preprocess(ls_val_str):
        ls_val_str = ls_val_str.replace(" ", "")
        # a compatible function in case user inputs double side bracket for list values
        if len(ls_val_str) >= 2 and ls_val_str[0] == "[" and ls_val_str[-1] == "]":
            return ls_val_str[1:-1]
        return ls_val_str
        
    # Python built-in values
    if val_type in ["int", "int32", "int64"]:
        return int(val_str)
    elif val_type in ["int[]", "int32[]", "int64[]"]:
        attr_val = []
        for v in preprocess(val_str).split(","):
            attr_val.append(int(v))
        return attr_val
    elif val_type in ["float", "float32", "float64"]:
        return float(val_str)
    elif val_type in ["float[]", "float32[]", "float64[]"]:
        attr_val = []
        for v in preprocess(val_str).split(","):
            attr_val.append(float(v))
        return attr_val
    elif val_type in ["string[]"]:
        attr_val = []
        for v in preprocess(val_str).split(","):
            attr_val.append(str(v))
        return attr_val
    
    # onnx built-in values 
    elif val_type == "DataType":
        # https://github.com/onnx/onnx/blob/46b96275554b0d978dd5c8ba786cc81dabd1d25a/onnx/onnx.proto#L479
        return getattr(TensorProto, val_str.upper())
        
    else:
        raise RuntimeError("type {} is not considered in current version.\n \
                            Current supported types are: int, int32, int64, int[], int32[], int64[], \
                            float, float32, float64 and float[], float32[], float64[].\n \
                            You are kindly to report an issue for this problem. Thanks!".format(val_type))


# map np datatype to onnx datatype
# https://github.com/onnx/onnx/blob/8669fad0247799f4d8683550eec749974b4f5338/onnx/helper.py#L1177
def np2onnxdtype(np_dtype):
    return cast(int, NP_TYPE_TO_TENSOR_TYPE[np_dtype])

if __name__ == "__main__":
    
    def tmp_debug():
        val = 0.0171247538316637
        np_fp32 = np.array(val, dtype=np.float32)
        np_fp64 = np.array(val, dtype=np.float64)
        np_fp64_conv = np.array(np_fp32, dtype=np.float64)

        print(val)
        np.set_printoptions(precision=20)
        print(np_fp32)
        print(np_fp64)
        print(np_fp32 == np_fp64)
        
        print(np_fp64_conv)
        
        pass
    # tmp_debug()
    
    def test_parse_str2np():
        # # tensor_str = "1.223"
        # tensor_str = "0.023"
        # val = parse_str2np(tensor_str, "float32")
        # print(type(val), val)
        
        # # tensor_str = "[1, 2, 3]"
        # tensor_str = "[[10, 2.3, 3],[1, 2e6, 3]]"
        # val = parse_str2np(tensor_str, "float32")
        # print(type(val), val)
        
        # tensor_str = "[[10, 2, 3],[1, 2, 3]]"
        # val = parse_str2np(tensor_str, "int64")
        # print(type(val), val)
        
        init_val_str = '[[[[0.0171247538316637]],[[0.0175070028011204]],[[0.0174291938997821]]]]'
        init_type = 'float32'
        init_val = parse_str2np(init_val_str, init_type)
        # print(init_val)
        pass
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
    # test_parse_str2val()
    
    