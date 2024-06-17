import numpy as np
from typing import cast
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx import TensorProto

# parse numpy values from string
def str2np(tensor_str, tensor_type):
    def parse_value(value_str, value_type):
        if value_type.startswith('int') or value_type.startswith('uint'):
            return int(value_str)
        elif value_type.startswith('float'):
            return float(value_str)
        else:
            raise RuntimeError("type {} is not considered in current version.\n \
                                Please report an issue for this problem. Thanks!".format(value_type))

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
    tensor_str = tensor_str.replace("\t", "")
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
        raise RuntimeError("Parse tensor fails!" + \
                          f"tensor type: {tensor_type}" + \
                          f"tensor value: {val}")

# parse Python or onnx built-in values from string
def str2val(val_str, val_type):
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
    elif val_type == "string":
        return str(val_str)
    elif val_type == "string[]":
        attr_val = []
        for v in preprocess(val_str).split(","):
            attr_val.append(str(v))
        return attr_val

    # onnx built-in values
    elif val_type == "DataType":
        # https://github.com/onnx/onnx/blob/46b96275554b0d978dd5c8ba786cc81dabd1d25a/onnx/onnx.proto#L479
        return getattr(TensorProto, val_str.upper())

    else:
        raise RuntimeError(f"type {val_type} is not considered in current version.\n" + \
                            "Currently supported types are:\n" + \
                            " - int, int32, int64, int[], int32[], int64[]\n" + \
                            " - float, float32, float64 and float[], float32[], float64[]\n" + \
                            " - string, string[]\n" + \
                            "Please report an issue for this problem. Thanks!")

# map np datatype to onnx datatype
# https://github.com/onnx/onnx/blob/8669fad0247799f4d8683550eec749974b4f5338/onnx/helper.py#L1177
def np2onnxdtype(np_dtype):
    return cast(int, NP_TYPE_TO_TENSOR_TYPE[np_dtype])

def str2onnxdtype(str_dtype):
    STR_TYPE_TO_TENSOR_TYPE = {
        'float32' : int(TensorProto.FLOAT),
        'uint8'   : int(TensorProto.UINT8),
        'int8'    : int(TensorProto.INT8),
        'uint16'  : int(TensorProto.UINT16),
        'int16' : int(TensorProto.INT16),
        'int32' : int(TensorProto.INT32),
        'int64' : int(TensorProto.INT64),
        'bool' : int(TensorProto.BOOL),
        'float16': int(TensorProto.FLOAT16),
        'bfloat16' : int(TensorProto.BFLOAT16),
        'float64': int(TensorProto.DOUBLE),
        'complex64': int(TensorProto.COMPLEX64),
        'complex128': int(TensorProto.COMPLEX128),
        'uint32' : int(TensorProto.UINT32),
        'uint64' : int(TensorProto.UINT64),
        'string': int(TensorProto.STRING)
    }
    return STR_TYPE_TO_TENSOR_TYPE[str_dtype]

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

    def test_str2np():
        # # tensor_str = "1.223"
        # tensor_str = "0.023"
        # val = str2np(tensor_str, "float32")
        # print(type(val), val)

        # # tensor_str = "[1, 2, 3]"
        # tensor_str = "[[10, 2.3, 3],[1, 2e6, 3]]"
        # val = str2np(tensor_str, "float32")
        # print(type(val), val)

        # tensor_str = "[[10, 2, 3],[1, 2, 3]]"
        # val = str2np(tensor_str, "int64")
        # print(type(val), val)

        init_val_str = '[[[[0.0171247538316637]],[[0.0175070028011204]],[[0.0174291938997821]]]]'
        init_type = 'float32'
        init_val = str2np(init_val_str, init_type)
        # print(init_val)
        pass
    test_str2np()

    def test_str2val():
        val_str = "1"
        val = str2val(val_str, "int")
        print(val)

        # val_str = "1, 2, 3"
        val_str = "[1, 2, 3]"
        val = str2val(val_str, "int[]")
        print(val)
        val = str2val(val_str, "float[]")
        print(val)

        val_str = "int8"
        # val_str = "float"
        val = str2val(val_str, "DataType")
        print(val)
    # test_str2val()

