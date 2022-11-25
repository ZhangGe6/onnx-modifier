# https://leimao.github.io/blog/ONNX-Python-API/
# https://leimao.github.io/blog/ONNX-IO-Stream/
# https://github.com/saurabh-shandilya/onnx-utils
# https://stackoverflow.com/questions/52402448/how-to-read-individual-layers-weight-bias-values-from-onnx-model

import os
import copy
import struct
import warnings
import numpy as np
import onnx
from onnx import numpy_helper
from utils import make_new_node, make_attr_changed_node
from utils import parse_str2np, np2onnxdtype

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
        print("loading model...")
        stream.seek(0)
        model_proto = onnx.load_model(stream, onnx.ModelProto, load_external_data=False)
        print("load done!")
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
    
    def change_batch_size(self, rebatch_info):
        if not (rebatch_info): return
        # https://github.com/onnx/onnx/issues/2182
        rebatch_type = rebatch_info['type']
        rebatch_value = rebatch_info['value']
        if rebatch_type == 'fixed':
            rebatch_value = int(rebatch_value)
        # print(rebatch_type, rebatch_value)
        
        # Change batch size in input, output and value_info
        for tensor in list(self.graph.input) + list(self.graph.value_info) + list(self.graph.output):
            if type(rebatch_value) == str:
                tensor.type.tensor_type.shape.dim[0].dim_param = rebatch_value
            elif type(rebatch_value) == int:
                tensor.type.tensor_type.shape.dim[0].dim_value = rebatch_value
            else:
                warnings.warn('Unknown type {} for batch size. Fallback to dynamic batch size.'.format(type(rebatch_value)))
                tensor.type.tensor_type.shape.dim[0].dim_param = str(rebatch_value)
        # print(type(rebatch_value), self.graph.input[0].type.tensor_type.shape.dim[0].dim_value)
        # print(type(rebatch_value), self.graph.input[0].type.tensor_type.shape.dim[0].dim_param)
        
        # handle reshapes
        for node in self.graph.node:
            if node.op_type != 'Reshape':
                continue
            for init in self.graph.initializer:
                # node.input[1] is expected to be a reshape
                if init.name != node.input[1]:
                    continue
                
                v = rebatch_value if rebatch_type == 'fixed' else -1
                # Shape is stored as a list of ints
                if len(init.int64_data) > 0:
                    # This overwrites bias nodes' reshape shape but should be fine
                    init.int64_data[0] = v
                # Shape is stored as bytes
                elif len(init.raw_data) > 0:
                    shape = bytearray(init.raw_data)
                    struct.pack_into('q', shape, 0, v)
                    init.raw_data = bytes(shape)
            
    def remove_node_by_name(self, node_name):
        # remove node in graph
        self.graph.node.remove(self.node_name2module[node_name])        
    
    def remove_model_output_by_name(self, node_name):
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
                    self.remove_model_output_by_name(node_name)
                else:
                    # print('removing node {} ...'.format(node_name))
                    self.remove_node_by_name(node_name)
        
        remained_node_inputs = []
        for remained_node in self.graph.node:
            remained_node_inputs += remained_node.input
        
        # remove node initializers (parameters), aka, keep and only keep the initializers of remained nodes
        for init_name in self.initializer_name2module.keys():
            if not init_name in remained_node_inputs:
                self.initializer.remove(self.initializer_name2module[init_name])

        # remove the (model) inputs related to deleted nodes 
        # https://github.com/ZhangGe6/onnx-modifier/issues/12
        for input_name in self.graph_input_names:
            if not input_name in remained_node_inputs:
                self.graph.input.remove(self.node_name2module[input_name])
                
    def modify_node_io_name(self, node_renamed_io):
        for node_name in node_renamed_io.keys():
            if node_name not in self.node_name2module.keys():
                # custom added nodes or custom added model outputs, or the deleted nodes
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
        
                    # rename the corresponding initializer and update initializer_name2module
                    if src_name in self.initializer_name2module.keys():
                        init = self.initializer_name2module[src_name]
                        init.name = dst_name
                        self.initializer_name2module[dst_name] = init
                        del self.initializer_name2module[src_name]
        
    def modify_node_attr(self, node_changed_attr):
        # we achieve it by deleting the original node and make a (copied) new node
        # print(node_changed_attr)
        for node_name in node_changed_attr.keys():
            orig_node = self.node_name2module[node_name]
            attr_changed_node = make_attr_changed_node(orig_node, node_changed_attr[node_name])
            self.graph.node.remove(self.node_name2module[node_name]) 
            self.graph.node.append(attr_changed_node)
            
            # update the node_name2module
            del self.node_name2module[node_name]
            self.node_name2module[node_name] = attr_changed_node
            
    def add_nodes(self, nodes_info, node_states):
        for node_info in nodes_info.values():
            if node_states[node_info['properties']['name']] == "Deleted":
                continue
            # print(node_info)
            node = make_new_node(node_info)
            # print(node)
            
            self.graph.node.append(node)
            
            # update the node_name2module
            self.node_name2module[node.name] = node

    def add_outputs(self, added_outputs):
        # https://github.com/onnx/onnx/issues/3277#issuecomment-1050600445
        added_output_names = added_outputs.values()
        if len(added_output_names) == 0: return 
        # filter out the deleted custom-added outputs
        value_info_protos = []
        shape_info = onnx.shape_inference.infer_shapes(self.model_proto)
        # print(added_output_names)
        for value_info in shape_info.graph.value_info:
            # print(value_info.name)
            if value_info.name in added_output_names:
                value_info_protos.append(value_info)
        self.graph.output.extend(value_info_protos)

    def modify_initializer(self, changed_initializer):
        # print(changed_initializer)
        for init_name, meta in changed_initializer.items():
            # https://github.com/onnx/onnx/issues/2978
            init_type, init_val_str = meta
            if init_val_str == "": continue # in case we clear the input
            # print(init_name, init_type, init_val)
            init_val = parse_str2np(init_val_str, init_type)
            # print(init_val)
            # for primary initilizers
            if init_name in self.initializer_name2module.keys():
                tensor = numpy_helper.from_array(init_val, init_name)
                self.initializer_name2module[init_name].CopyFrom(tensor)
            # for custom added initilizers
            else:
                # more details about why the .flatten() is needed can be seen in https://github.com/ZhangGe6/onnx-modifier/issues/28
                init_val_flat = init_val
                if len(init_val.shape) > 1:
                    init_val_flat = init_val.flatten()
                initializer_tensor = onnx.helper.make_tensor(
                    name=init_name,
                    data_type=np2onnxdtype(init_val.dtype),
                    dims=init_val.shape,
                    vals=init_val_flat)
                # print(initializer_tensor)
                self.initializer.append(initializer_tensor)
                self.initializer_name2module[init_name] = initializer_tensor
        
    def post_process(self):
        def remove_isolated_nodes():
            # remove the remained corresponding isolated nodes, like Constant
            remained_node_inputs, remained_node_outputs = [], []
            for remained_node in self.graph.node:
                remained_node_inputs += remained_node.input
                remained_node_outputs += remained_node.output
                
            for remained_node in self.graph.node:
                # delete the node if it does not serve as the input or output of any other nodes
                unused = True
                for output in remained_node.output:
                    if output in remained_node_inputs:
                        unused = False
                        break
                for input in remained_node.input:
                    if input in remained_node_outputs:
                        unused = False
                        break    
                if unused:
                    self.graph.node.remove(self.node_name2module[remained_node.name])
                    for inp in remained_node.input:
                        if inp in self.initializer_name2module.keys():
                            self.initializer.remove(self.initializer_name2module[inp])
                            
        remove_isolated_nodes()
                        
    def modify(self, modify_info):
        '''
        1. Some functions, such as modify_initializer(), should be placed 
        before modify_node_io_name(), to avoid name mismatch error.
        2. add_nodes() should be placed at the first place, otherwise
        remove_node_by_node_states() will delete the initializer of 
        newly added nodes mistakenly
        '''
        # print(modify_info['node_states'])
        # print(modify_info['node_renamed_io'])
        # print(modify_info['node_changed_attr'])
        # print(modify_info['added_node_info'])
        # print(modify_info['added_outputs'])  
        
        self.add_nodes(modify_info['added_node_info'], modify_info['node_states'])
        self.modify_initializer(modify_info['changed_initializer'])
        self.change_batch_size(modify_info['rebatch_info'])
        self.remove_node_by_node_states(modify_info['node_states'])
        self.modify_node_io_name(modify_info['node_renamed_io'])
        self.modify_node_attr(modify_info['node_changed_attr'])
        self.add_outputs(modify_info['added_outputs'])
        
        self.post_process()

    def check_and_save_model(self, save_dir='./modified_onnx'):
        print("saving model...")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'modified_' + self.model_name)
        
        # adding new node like self.add_nodes() and self.modify_node_attr() can not guarantee the nodes are topologically sorted
        # so `onnx.onnx_cpp2py_export.checker.ValidationError: Nodes in a graph must be topologically sorted` will be invoked
        # I turn off the onnx checker as a workaround.
        # onnx.checker.check_model(self.model_proto)
        onnx.save(self.model_proto, save_path)
        print("model saved in {} !".format(save_dir))
    
    def inference(self, input_shape=[1, 3, 224, 224], x=None, output_names=None):
        import onnxruntime as rt
        model_proto_bytes = onnx._serialize(self.model_proto)
        inference_session = rt.InferenceSession(model_proto_bytes)
        
        if not x:
            np.random.seed(0)
            x = np.random.randn(*input_shape).astype(np.float32)
        if not output_names:
            output_name = self.graph.node[-1].output[0]
            # output_value_info = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.INT64, shape=[])
            output_value_info = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, shape=[])
            self.graph.output.append(output_value_info)
            output_names = [inference_session.get_outputs()[0].name]
        
        input_name = inference_session.get_inputs()[0].name
        out = inference_session.run(output_names, {input_name: x})[0]
        print(out.shape, out.dtype)
        # print(out[0][0][0][0])