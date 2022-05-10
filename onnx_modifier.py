# https://leimao.github.io/blog/ONNX-Python-API/
# https://leimao.github.io/blog/ONNX-IO-Stream/
# https://github.com/saurabh-shandilya/onnx-utils
# https://stackoverflow.com/questions/52402448/how-to-read-individual-layers-weight-bias-values-from-onnx-model

import os
import copy
import onnx

class onnxModifier:
    def __init__(self, model_name, model_proto):
        self.model_name = model_name
        self.model_proto_backup = model_proto
        self.reload()

    @classmethod
    def from_model_path(cls, model_path):
        model_name = os.path.basename(model_path)
        model_proto = onnx.load(model_path)
        return cls(model_name, model_proto)

    @classmethod
    def from_name_stream(cls, name, stream):
        # https://leimao.github.io/blog/ONNX-IO-Stream/
        stream.seek(0)
        model_proto = onnx.load_model(stream, onnx.ModelProto, load_external_data=False)
        return cls(name, model_proto)

    def reload(self):
        self.model_proto = copy.deepcopy(self.model_proto_backup)
        self.graph = self.model_proto.graph
        self.initializer = self.model_proto.graph.initializer
        
        self.gen_name2module_map()
    
    def gen_name2module_map(self):
        # node name => node
        self.node_name2module = dict()
        node_idx = 0
        for node in self.graph.node:
            if node.name == '':
                node.name = str(node.op_type) + str(node_idx)
            node_idx += 1
            self.node_name2module[node.name] = node
            
        for out in self.graph.output:
            self.node_name2module["out_" + out.name] = out  # add `out_` in case the output has the same name with the last node
        self.graph_output_names = ["out_" + out.name for out in self.graph.output]
        # print(self.node_name2module.keys())
        
        # initializer name => initializer
        self.initilizer_name2module = dict()
        for initializer in self.initializer:
            self.initilizer_name2module[initializer.name] = initializer
                   
    def remove_node_by_name(self, node_name):
        # remove node in graph
        self.graph.node.remove(self.node_name2module[node_name])        
    
    def remove_output_by_name(self, node_name):
        self.graph.output.remove(self.node_name2module[node_name])

    def remove_node_by_node_states(self, node_states):
        # remove node from graph
        for node_name, node_state in node_states.items():
            if node_state == 'Deleted':
                if node_name in self.graph_output_names:
                    # print('removing output {} ...'.format(node_name))
                    self.remove_output_by_name(node_name)
                else:
                    # print('removing node {} ...'.format(node_name))
                    self.remove_node_by_name(node_name)
        
        # remove node initializers (parameters) aka, keep and only keep the initializers of left nodes
        left_node_inputs = []
        for left_node in self.graph.node:
            left_node_inputs += left_node.input
        
        for init_name in self.initilizer_name2module.keys():
            if not init_name in left_node_inputs:
                self.initializer.remove(self.initilizer_name2module[init_name])
    
    def modify_node_io_name(self, node_renamed_io):
        # print(node_renamed_io)
        for node_name in node_renamed_io.keys():
            renamed_ios = node_renamed_io[node_name]
            for src_name, dst_name in renamed_ios.items():
                # print(src_name, dst_name)
                node = self.node_name2module[node_name]
                # print(node.input, node.output)
                for i in range(len(node.input)):
                    if node.input[i] == src_name:
                        node.input[i] = dst_name
                for i in range(len(node.output)):
                    if node.output[i] == src_name:
                        node.output[i] = dst_name    
                # print(node.input, node.output)
            
    def check_and_save_model(self, save_dir='./modified_onnx'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        save_path = os.path.join(save_dir, 'modified_' + self.model_name)
        onnx.checker.check_model(self.model_proto)
        onnx.save(self.model_proto, save_path)  
    
    def inference(self):
        # model_proto_bytes = onnx._serialize(model_proto_from_stream)
        # inference_session = rt.InferenceSession(model_proto_bytes)
        pass
        
        
if __name__ == "__main__":
    model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-3.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-12-int8.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\tflite_sim.onnx"
    onnx_modifier = onnxModifier.from_model_path(model_path)
        
    def remove_node_by_node_states():
        print(len(onnx_modifier.graph.node))
        print(len(onnx_modifier.graph.initializer))
        node_states_fp = {'data_0': 'Exist', 'Conv0': 'Exist', 'Relu1': 'Exist', 'MaxPool2': 'Exist', 'Conv3': 'Exist', 'Relu4': 'Exist', 'Conv5': 'Exist', 'Relu6': 'Exist', 'Conv7': 'Deleted', 'Relu8': 'Deleted', 'Concat9': 'Deleted', 'Conv10': 'Deleted', 'Relu11': 'Deleted', 'Conv12': 'Deleted', 'Relu13': 'Deleted', 'Conv14': 'Deleted', 'Relu15': 'Deleted', 'Concat16': 'Deleted', 'MaxPool17': 'Deleted', 'Conv18': 'Deleted', 'Relu19': 'Deleted', 'Conv20': 'Deleted', 'Relu21': 'Deleted', 'Conv22': 'Deleted', 'Relu23': 'Deleted', 'Concat24': 'Deleted', 'Conv25': 'Deleted', 'Relu26': 'Deleted', 'Conv27': 'Deleted', 'Relu28': 'Deleted', 'Conv29': 'Deleted', 'Relu30': 'Deleted', 'Concat31': 'Deleted', 'MaxPool32': 'Deleted', 'Conv33': 'Deleted', 'Relu34': 'Deleted', 'Conv35': 'Deleted', 'Relu36': 'Deleted', 'Conv37': 'Deleted', 'Relu38': 'Deleted', 'Concat39': 'Deleted', 'Conv40': 'Deleted', 'Relu41': 'Deleted', 'Conv42': 'Deleted', 'Relu43': 'Deleted', 'Conv44': 'Deleted', 'Relu45': 'Deleted', 'Concat46': 'Deleted', 'Conv47': 'Deleted', 'Relu48': 'Deleted', 'Conv49': 'Deleted', 'Relu50': 'Deleted', 'Conv51': 'Deleted', 'Relu52': 'Deleted', 'Concat53': 'Deleted', 'Conv54': 'Deleted', 'Relu55': 'Deleted', 'Conv56': 'Deleted', 'Relu57': 'Deleted', 'Conv58': 'Deleted', 'Relu59': 'Deleted', 'Concat60': 'Deleted', 'Dropout61': 'Deleted', 'Conv62': 'Deleted', 'Relu63': 'Deleted', 'GlobalAveragePool64': 'Deleted', 'Softmax65': 'Deleted', 'softmaxout_1': 'Deleted'}
        
        node_states_quant = {'data_0': 'Exist', 'data_0_QuantizeLinear': 'Exist', 'Conv_nc_rename_0_quant': 'Exist', 'MaxPool_nc_rename_2_quant': 'Exist', 'Conv_nc_rename_3_quant': 'Deleted', 'Conv_nc_rename_5_quant': 'Deleted', 'Conv_nc_rename_7_quant': 'Deleted', 'fire2/expand1x1_2_DequantizeLinear': 'Deleted', 'fire2/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_9': 'Deleted', 'fire2/concat_1_QuantizeLinear': 'Deleted', 'Conv_nc_rename_10_quant': 'Deleted', 'Conv_nc_rename_12_quant': 'Deleted', 'Conv_nc_rename_14_quant': 'Deleted', 'fire3/expand1x1_2_DequantizeLinear': 'Deleted', 
        'fire3/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_16': 'Deleted', 'MaxPool_nc_rename_17': 'Deleted', 'pool3_1_QuantizeLinear': 
        'Deleted', 'Conv_nc_rename_18_quant': 'Deleted', 'Conv_nc_rename_20_quant': 'Deleted', 'Conv_nc_rename_22_quant': 'Deleted', 'fire4/expand1x1_2_DequantizeLinear': 'Deleted', 'fire4/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_24': 'Deleted', 'fire4/concat_1_QuantizeLinear': 'Deleted', 'Conv_nc_rename_25_quant': 'Deleted', 'Conv_nc_rename_27_quant': 'Deleted', 'Conv_nc_rename_29_quant': 'Deleted', 'fire5/expand1x1_2_DequantizeLinear': 'Deleted', 'fire5/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_31': 'Deleted', 'MaxPool_nc_rename_32': 'Deleted', 'pool5_1_QuantizeLinear': 'Deleted', 'Conv_nc_rename_33_quant': 'Deleted', 'Conv_nc_rename_35_quant': 'Deleted', 'Conv_nc_rename_37_quant': 'Deleted', 'fire6/expand1x1_2_DequantizeLinear': 'Deleted', 'fire6/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_39': 'Deleted', 'fire6/concat_1_QuantizeLinear': 'Deleted', 'Conv_nc_rename_40_quant': 'Deleted', 'Conv_nc_rename_42_quant': 'Deleted', 'Conv_nc_rename_44_quant': 'Deleted', 'fire7/expand1x1_2_DequantizeLinear': 'Deleted', 'fire7/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_46': 'Deleted', 'fire7/concat_1_QuantizeLinear': 'Deleted', 'Conv_nc_rename_47_quant': 'Deleted', 'Conv_nc_rename_49_quant': 'Deleted', 'Conv_nc_rename_51_quant': 'Deleted', 'fire8/expand1x1_2_DequantizeLinear': 'Deleted', 'fire8/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_53': 'Deleted', 'fire8/concat_1_QuantizeLinear': 'Deleted', 'Conv_nc_rename_54_quant': 'Deleted', 'Conv_nc_rename_56_quant': 'Deleted', 'Conv_nc_rename_58_quant': 'Deleted', 'fire9/expand1x1_2_DequantizeLinear': 'Deleted', 'fire9/expand3x3_2_DequantizeLinear': 'Deleted', 'Concat_nc_rename_60': 'Deleted', 'fire9/concat_1_QuantizeLinear': 'Deleted', 'Conv_nc_rename_61_quant': 'Deleted', 'GlobalAveragePool_nc_rename_63_quant': 'Deleted', 'pool10_1_DequantizeLinear': 'Deleted', 'Softmax_nc_rename_64': 'Deleted', 'softmaxout_1': 'Deleted'}

        
        node_states = node_states_quant
        # node_states = node_states_fp
        # print('\graph  input')
        # for inp in onnx_modifier.graph.input:
        #     print(inp.name)
        onnx_modifier.remove_node_by_node_states(node_states)
        print(len(onnx_modifier.graph.node))
        print(len(onnx_modifier.graph.initializer))
        print(len(onnx_modifier.initilizer_name2module.keys()))
        # print(onnx_modifier.initilizer_name2module.keys())
        # for i, k in enumerate(onnx_modifier.initilizer_name2module.keys()):
        #     print("\nremoving", i, k)
        #     onnx_modifier.graph.initializer.remove(onnx_modifier.initilizer_name2module[k])
        #     print("removed")
            
        print('\nleft initializers:')
        for initializer in onnx_modifier.model_proto.graph.initializer:
            print(initializer.name)
        
        print('\nleft nodes:')
        for node in onnx_modifier.graph.node:
            print(node.name)
        
        print('\nleft input')
        for inp in onnx_modifier.graph.input:
            print(inp.name)
        
        onnx_modifier.check_and_save_model()
    # remove_node_by_node_states()
    
    def explore_basic():
        print(type(onnx_modifier.model_proto.graph.initializer))
        print(dir(onnx_modifier.model_proto.graph.initializer))
        
        print(len(onnx_modifier.model_proto.graph.node))
        print(len(onnx_modifier.model_proto.graph.initializer))
        
        for node in onnx_modifier.model_proto.graph.node:
            print(node.name)
            print(node.input)
            print()
            
        # for initializer in onnx_modifier.model_proto.graph.initializer:
        #     print(initializer.name)
        # print(onnx_modifier.model_proto.graph.initializer['fire9/concat_1_scale'])
        pass    
    # explore_basic()
    
    def test_modify_node_io_name():
        node_rename_io = {'Conv3': {'pool1_1': 'conv1_1'}}
        onnx_modifier.modify_node_io_name(node_rename_io)
        onnx_modifier.check_and_save_model()      
    test_modify_node_io_name()
        