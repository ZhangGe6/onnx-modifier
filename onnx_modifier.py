# https://leimao.github.io/blog/ONNX-Python-API/
# https://leimao.github.io/blog/ONNX-IO-Stream/
# https://github.com/saurabh-shandilya/onnx-utils
# https://stackoverflow.com/questions/52402448/how-to-read-individual-layers-weight-bias-values-from-onnx-model

import os
import copy
import numpy as np
import onnx
from utils import make_new_node, make_attr_changed_node

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
        
        for inp in self.graph.input:
            self.node_name2module[inp.name] = inp
        self.graph_input_names = [inp.name for inp in self.graph.input]
            
        for out in self.graph.output:
            self.node_name2module["out_" + out.name] = out  # add `out_` in case the output has the same name with the last node
        self.graph_output_names = ["out_" + out.name for out in self.graph.output]
        # print(self.node_name2module.keys())
        
        # initializer name => initializer
        self.initializer_name2module = dict()
        for initializer in self.initializer:
            self.initializer_name2module[initializer.name] = initializer
                   
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
        
        # remove node initializers (parameters), aka, keep and only keep the initializers of left nodes
        left_node_inputs = []
        for left_node in self.graph.node:
            left_node_inputs += left_node.input
        
        for init_name in self.initializer_name2module.keys():
            if not init_name in left_node_inputs:
                self.initializer.remove(self.initializer_name2module[init_name])

        # remove the (model) inputs related to deleted nodes 
        # https://github.com/ZhangGe6/onnx-modifier/issues/12
        for input_name in self.graph_input_names:
            if not input_name in left_node_inputs:
                self.graph.input.remove(self.node_name2module[input_name])
        
        # remove the left unused Constant nodes
        for left_node in self.graph.node:
            if left_node.op_type == "Constant":
                output_deleted = [False] * len(left_node.output)
                for i, output in enumerate(left_node.output):
                    if not (output in left_node_inputs):
                        output_deleted[i] = True

                const_node_left_output = [left_node.output[i] for i in range(len(left_node.output)) if not output_deleted[i]]
                if len(const_node_left_output) == 0:
                    self.graph.node.remove(self.node_name2module[left_node.name]) 
                    # self.initializer.remove(self.initializer_name2module[init_name])
            
    def modify_node_io_name(self, node_renamed_io):
        for node_name in node_renamed_io.keys():
            if node_name not in self.node_name2module.keys():
                # custom added nodes or custom added model outputs
                continue
            renamed_ios = node_renamed_io[node_name]
            for src_name, dst_name in renamed_ios.items():
                node = self.node_name2module[node_name]
                if node_name in self.graph_input_names:
                    node.name = dst_name
                elif node_name in self.graph_output_names:
                    node.name = dst_name
                else:
                    # print(node.input, node.output)
                    for i in range(len(node.input)):
                        if node.input[i] == src_name:
                            node.input[i] = dst_name
                    for i in range(len(node.output)):
                        if node.output[i] == src_name:
                            node.output[i] = dst_name   
        
    def modify_node_attr(self, node_changed_attr):
        # we achieve it by deleting the original node and make a (copied) new node
        # print(node_changed_attr)
        for node_name in node_changed_attr.keys():
            orig_node = self.node_name2module[node_name]
            attr_changed_node = make_attr_changed_node(orig_node, node_changed_attr[node_name])
            self.graph.node.remove(self.node_name2module[node_name]) 
            self.graph.node.append(attr_changed_node)
            
        # update the node_name2module and initializer_name2module
        self.gen_name2module_map()            
            
    def add_node(self, nodes_info, node_states):
        for node_info in nodes_info.values():
            if node_states[node_info['properties']['name']] == "Deleted":
                continue
            node = make_new_node(node_info)
            # print(node)
            
            self.graph.node.append(node)

    def add_outputs(self, added_outputs, node_states):
        # https://github.com/onnx/onnx/issues/3277#issuecomment-1050600445
        added_output_names = added_outputs.values()
        # filter out the deleted custom-added outputs
        value_info_protos = []
        shape_info = onnx.shape_inference.infer_shapes(self.model_proto)
        for value_info in shape_info.graph.value_info:
            if value_info.name in added_output_names:
                value_info_protos.append(value_info)
        self.graph.output.extend(value_info_protos)

    def modify(self, modify_info):
        # print(modify_info['node_states'])
        # print(modify_info['node_renamed_io'])
        # print(modify_info['node_changed_attr'])
        # print(modify_info['added_node_info'])
        # print(modify_info['added_outputs'])
        self.remove_node_by_node_states(modify_info['node_states'])
        self.modify_node_io_name(modify_info['node_renamed_io'])
        self.modify_node_attr(modify_info['node_changed_attr'])
        self.add_node(modify_info['added_node_info'], modify_info['node_states'])
        self.add_outputs(modify_info['added_outputs'], modify_info['node_states'])
          
    
    def check_and_save_model(self, save_dir='./modified_onnx'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'modified_' + self.model_name)
        
        # adding new node like self.add_node() and self.modify_node_attr() can not guarantee the nodes are topologically sorted
        # so `onnx.onnx_cpp2py_export.checker.ValidationError: Nodes in a graph must be topologically sorted` will be invoked
        # I turn off the onnx checker as a workaround.
        # onnx.checker.check_model(self.model_proto)
        onnx.save(self.model_proto, save_path)  
    
    def inference(self, x=None, output_names=None):
        import onnxruntime as rt     
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

        # This issue may be encountered: https://github.com/microsoft/onnxruntime/issues/7506
        out = inference_session.run(None, {input_name: x})[0]
        print(out.shape)
        
if __name__ == "__main__":
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-3.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-12-int8.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\tflite_sim.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-12.onnx"
    model_path = "C:\\Users\\ZhangGe\\Desktop\\with-added-output-modified_modified_squeezenet1.0-12.onnx"
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-3.onnx"   // TODO: this model is not supported well , but why?
    # model_path = "C:\\Users\\ZhangGe\\Desktop\\mobilenetv2-7.onnx"
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
        # print(len(onnx_modifier.graph.node))
        # print(len(onnx_modifier.graph.initializer))

        # print(onnx_modifier.node_name2module.keys())
        # print(onnx_modifier.graph.node)
        # for node in onnx_modifier.graph.node:
        #     print(node.name)
        #     print(node.input)
        #     print(node.output)
        
        
        node_states = {'input': 'Exist', 'Conv_0': 'Exist', 'Conv_95': 'Exist', 'Clip_96': 'Deleted', 'GlobalAveragePool_97': 'Deleted', 'Shape_98': 'Deleted', 'Gather_100': 'Deleted', 'Unsqueeze_101': 'Deleted', 'Concat_102': 'Deleted', 'Reshape_103': 'Deleted', 'Gemm_104': 'Deleted', 'out_output': 'Deleted'}
        # print('\graph  input')
        # for inp in onnx_modifier.graph.input:
        #     print(inp.name)
        onnx_modifier.remove_node_by_node_states(node_states)
        # print(len(onnx_modifier.graph.node))
        # print(len(onnx_modifier.graph.initializer))
        # print(len(onnx_modifier.initializer_name2module.keys()))

        for node in onnx_modifier.graph.node:
            print(node.name)
            print(node.input, node.output)
        for initializer in onnx_modifier.initializer:
            print(initializer.name)
    
        
        # print('\nleft nodes:')
        # for node in onnx_modifier.graph.node:
        #     print(node.name)
        
        # print('\nleft input')
        # for inp in onnx_modifier.graph.input:
        #     print(inp.name)
        
        onnx_modifier.check_and_save_model()
    # remove_node_by_node_states()

    def test_modify_node_io_name():
        node_rename_io = {'input': {'input': 'inputd'}, 'Conv_0': {'input': 'inputd'}}
        onnx_modifier.modify_node_io_name(node_rename_io)
        onnx_modifier.check_and_save_model()      
    # test_modify_node_io_name()

    def test_add_node():
        node_info = {'custom_added_AveragePool0': {'properties': {'domain': 'ai.onnx', 'op_type': 'AveragePool', 'name': 'custom_added_AveragePool0'}, 'attributes': {'kernel_shape': [2, 2]}, 'inputs': {'X': ['fire2/squeeze1x1_1']}, 'outputs': {'Y': ['out']}}}
    
        onnx_modifier.add_node(node_info)
        
        onnx_modifier.inference()
        onnx_modifier.check_and_save_model()  
    # test_add_node()

    def test_change_node_attr():
        # changed_attr = {'Clip_3': {'max': 5}}
        changed_attr = {'Conv_2': {'group': 64}}

        onnx_modifier.modify_node_attr(changed_attr)
        
        onnx_modifier.check_and_save_model()
    # test_change_node_attr()
        
    def test_inference():
        onnx_modifier.inference()
    # test_inference()
    
    def test_add_output():
        # print(onnx_modifier.graph.output)
        onnx_modifier.add_outputs(['fire2/squeeze1x1_1'])
        # print(onnx_modifier.graph.output)
        onnx_modifier.check_and_save_model()
    # test_add_output()