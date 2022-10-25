import numpy as np

def parse_value(value_str, value_type):
    if value_type.startswith('int'):
        return int(value_str)
    elif value_type.startswith('float'):
        return float(value_str)
    else:
        raise RuntimeError("type {} is not considered in current version. \
                            You can kindly report an issue for this problem. Thanks!".format(value_type))

def parse_tensor(tensor_str, tensor_type):
    def extract_val():
        num_str = ""
        while (len(stk) > 0) and (type(stk[-1]) == str and ord('0') <= ord(stk[-1]) <= ord('9') or stk[-1] in ['+', '-', '.', 'e', 'E']):
            num_str = stk.pop() + num_str
        
        if len(num_str) > 0:
            return parse_value(num_str, tensor_type)
        else:
            return None
    
    tensor_str = tensor_str.replace(" ", "")
    stk = []
    for i, c in enumerate(tensor_str): # '['  ','  ']' '.' '-' or value
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

if __name__ == "__main__":
    # tensor_str = "1"
    # tensor_str = "[1, 2, 3]"
    tensor_str = "[[10, 2.3, 3],[1, 2e6, 3]]"
    val = parse_tensor(tensor_str, "float32")
    print(type(val), val)
    
    tensor_str = "[[10, 2, 3],[1, 2, 3]]"
    val = parse_tensor(tensor_str, "int64")
    print(type(val), val)
    
    