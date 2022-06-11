# https://leimao.github.io/blog/ONNX-Python-API/
# https://leimao.github.io/blog/ONNX-IO-Stream/
# https://github.com/saurabh-shandilya/onnx-utils
# https://stackoverflow.com/questions/52402448/how-to-read-individual-layers-weight-bias-values-from-onnx-model

import os
import copy
import numpy as np
import onnx
import onnxruntime as rt
from utils import make_node

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
            if not (node_name in self.node_name2module):
                # for custom added node here
                continue
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
    
    
    def add_node(self, nodes_info):
        for node_info in nodes_info.values():
            node = make_node(node_info)
            # print(node)
            
            self.graph.node.append(node)

    
    def modify(self, modify_info):
        # print(modify_info['node_renamed_io'])
        print(modify_info['added_node_info'])
        self.remove_node_by_node_states(modify_info['node_states'])
        self.modify_node_io_name(modify_info['node_renamed_io'])  
        self.add_node(modify_info['added_node_info'])    
    
    def check_and_save_model(self, save_dir='./modified_onnx'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        save_path = os.path.join(save_dir, 'modified_' + self.model_name)
        onnx.checker.check_model(self.model_proto)
        onnx.save(self.model_proto, save_path)  
    
    def inference(self, x=None, output_names=None):        
        if not x:
            input_shape = [1, 3, 224, 224]
            x = np.random.randn(*input_shape).astype(np.float32)
        if not output_names:
            output_name = self.graph.node[-1].output[0]
            # output_value_info = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.INT64, shape=[])
            output_value_info = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape=[])
            self.graph.output.append(output_value_info)

        model_proto_bytes = onnx._serialize(self.model_proto)
        inference_session = rt.InferenceSession(model_proto_bytes)
        
        input_name = inference_session.get_inputs()[0].name
        output_name = inference_session.get_outputs()[0].name        
        # print(input_name)
        # print(output_name)
        
        # This issue may be encountered: https://github.com/microsoft/onnxruntime/issues/7506
        out = inference_session.run(None, {input_name: x})[0]

        # print(out)

        
if __name__ == "__main__":
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-3.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-12-int8.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\tflite_sim.onnx"
    model_path = "C:\\Users\\ZhangGe\\Desktop\\modified_modified_squeezenet1.0-12.onnx"
    onnx_modifier = onnxModifier.from_model_path(model_path)
    
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
   
    def remove_node_by_node_states():
        print(len(onnx_modifier.graph.node))
        print(len(onnx_modifier.graph.initializer))
        node_states_fp = {}
        node_states_quant = {}

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
    

    def test_modify_node_io_name():
        node_rename_io = {'Conv3': {'pool1_1': 'conv1_1'}}
        onnx_modifier.modify_node_io_name(node_rename_io)
        onnx_modifier.check_and_save_model()      
    # test_modify_node_io_name()

    def test_add_node():
        node_info = {'custom_added_AveragePool0': {'properties': {'domain': 'ai.onnx', 'op_type': 'AveragePool', 'name': 'custom_added_AveragePool0'}, 'attributes': {'kernel_shape': [2, 2]}, 'inputs': {'X': ['fire2/squeeze1x1_1']}, 'outputs': {'Y': ['out']}}}
    
        onnx_modifier.add_node(node_info)
        
        onnx_modifier.inference()
        onnx_modifier.check_and_save_model()  
        
    test_add_node()
        
        