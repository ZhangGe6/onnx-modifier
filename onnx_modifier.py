# https://leimao.github.io/blog/ONNX-Python-API/
# https://leimao.github.io/blog/ONNX-IO-Stream/
# https://github.com/saurabh-shandilya/onnx-utils
# https://stackoverflow.com/questions/52402448/how-to-read-individual-layers-weight-bias-values-from-onnx-model

import os
import copy
import struct
import warnings
import platform
import numpy as np
import onnx
from onnx import numpy_helper
from utils import parse_str2np, parse_str2val
from utils import np2onnxdtype, str2onnxdtype
from utils import make_new_node, make_attr_changed_node
from utils import get_infered_shape
import json
from google.protobuf.json_format import Parse, MessageToJson

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
    def from_model_json(cls, name, stream):
        print("loading model json..."+name)
        stream.seek(0)
        onnx_json = json.load(stream)
        onnx_str = json.dumps(onnx_json)
        model_proto = Parse(onnx_str, onnx.ModelProto())
        # onnx.save(model_proto, name[:-4]+"onnx")
        # print("save model ..."+name[:-4]+"onnx")
        return cls(name, model_proto), name[:-4]+"onnx"

    @classmethod
    def from_name_stream(cls, name, stream):
        # https://leimao.github.io/blog/ONNX-IO-Stream/
        print("loading model...")
        stream.seek(0)
        model_proto = onnx.load_model(stream, "protobuf", load_external_data=False)
        print("load done!")
        return cls(name, model_proto)

    def reload(self):
        self.model_proto = copy.deepcopy(self.model_proto_backup)
        self.graph = self.model_proto.graph
        self.initializer = self.model_proto.graph.initializer

        self.need_topsort = False
        self.gen_name2module_map()

    def gen_name2module_map(self):
        # node name => node
        self.cont_node_outname2name = dict()
        self.node_name2module = dict()
        node_idx = 0
        for node in self.graph.node:
            if node.name == '':
                node.name = str(node.op_type) + str(node_idx)
            node_idx += 1
            self.node_name2module[node.name] = node

            if node.op_type == "Constant":
                self.cont_node_outname2name[node.output[0]] = node.name

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
        # print(rebatch_type, rebatch_value)
        if rebatch_type == 'fixed':
            rebatch_value = int(rebatch_value)
            for tensor in self.graph.input:
                if type(rebatch_value) == str:
                    tensor.type.tensor_type.shape.dim[0].dim_param = rebatch_value
                elif type(rebatch_value) == int:
                    tensor.type.tensor_type.shape.dim[0].dim_value = rebatch_value
                else:
                    warnings.warn('Unknown type {} for batch size. Fallback to dynamic batch size.'.format(type(rebatch_value)))
                    tensor.type.tensor_type.shape.dim[0].dim_param = str(rebatch_value)

            self.shape_inference()
        else: # dynamic batch size
            # Change batch size in input, output and value_info
            for tensor in list(self.graph.input) + list(self.graph.value_info) + list(self.graph.output):
                tensor.type.tensor_type.shape.dim[0].dim_param = rebatch_value
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
                    # Shape is stored as a list of ints
                    if len(init.int64_data) > 0:
                        # This overwrites bias nodes' reshape shape but should be fine
                        init.int64_data[0] = -1
                    # Shape is stored as bytes
                    elif len(init.raw_data) > 0:
                        shape = bytearray(init.raw_data)
                        struct.pack_into('q', shape, 0, -1)
                        init.raw_data = bytes(shape)

    def remove_node_by_node_states(self, node_states):
        # remove node from graph
        for node_name, node_state in node_states.items():
            if not (node_name in self.node_name2module):
                # for custom added node here
                continue
            if node_state == 'Deleted':
                if node_name in self.graph_output_names:
                    # print('removing output {} ...'.format(node_name))
                    self.graph.output.remove(self.node_name2module[node_name])
                    self.graph_output_names = [n for n in self.graph_output_names if n != node_name]
                    self.node_name2module.pop(node_name, None)
                elif node_name not in self.graph_input_names:
                    # print('removing node {} ...'.format(node_name))
                    self.graph.node.remove(self.node_name2module[node_name])
                    self.node_name2module.pop(node_name, None)

        remained_inputs = []
        for remained_node in self.graph.node:
            remained_inputs += remained_node.input

        # remove node initializers (parameters), aka, keep and only keep the initializers of remained nodes
        for init_name in self.initializer_name2module.keys():
            if not init_name in remained_inputs:
                self.initializer.remove(self.initializer_name2module[init_name])

        # remove the (model) inputs related to deleted nodes
        # https://github.com/ZhangGe6/onnx-modifier/issues/12
        for input_name in self.graph_input_names:
            if not input_name in remained_inputs:
                self.graph.input.remove(self.node_name2module[input_name])
                self.node_name2module.pop(input_name, None)

        self.need_topsort = True

    def modify_node_io_name(self, node_renamed_io, added_inputs):
        for node_name in node_renamed_io.keys():
            if node_name not in self.node_name2module.keys():
                # added inputs
                if node_renamed_io[node_name][node_name] in [name_shape[0] for name_shape in added_inputs.values()]:
                    for key,val in self.node_name2module.items():
                        if hasattr(val,'input'):
                            for i in range(len(val.input)):
                                if val.input[i] == node_name:
                                    val.input[i] = node_renamed_io[node_name][node_name]
                # added inputs + custom added nodes or custom added model outputs, or the deleted nodes
                continue

            renamed_ios = node_renamed_io[node_name]
            for src_name, dst_name in renamed_ios.items():
                node = self.node_name2module[node_name]
                if node_name in self.graph_input_names:
                    node.name = dst_name
                    self.graph_input_names.remove(node_name)
                    self.graph_input_names.append(dst_name)
                    self.node_name2module[dst_name] = node
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
                    # else rename output of the corresponding constant node and update cont_node_outname2name
                    elif src_name in self.cont_node_outname2name.keys():
                        cont = self.node_name2module[self.cont_node_outname2name[src_name]]
                        cont.output[0] = dst_name
                        self.cont_node_outname2name[dst_name] = self.cont_node_outname2name[src_name]
                        del self.cont_node_outname2name[src_name]


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

        self.need_topsort = True

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
        self.need_topsort = True

    def add_inputs(self, inputs):
        for name_shape in inputs.values():
            # ['input.4', 'float32[1,8,96,96]']
            name = name_shape[0]
            dtype = name_shape[1].split("[")[0]
            onnx_dtype = str2onnxdtype(dtype)
            shape_str = name_shape[1].split("[")[1].split("]")[0]
            shape = parse_str2val(shape_str, "int[]")
            value_info = onnx.helper.make_tensor_value_info(
                                        name, onnx_dtype, shape)
            self.graph.input.append(value_info)
            self.graph_input_names.append(value_info.name)
            self.node_name2module[value_info.name] = value_info
    def modify_inputs(self, modefied_inputs):
        for name_shape in modefied_inputs.values():
            # ['input.4', 'float32[1,8,96,96]']
            name = name_shape[0]
            dtype = name_shape[1].split("[")[0]
            onnx_dtype = str2onnxdtype(dtype)
            shape_str = name_shape[1].split("[")[1].split("]")[0]
            shape = parse_str2val(shape_str, "int[]")
            value_info = onnx.helper.make_tensor_value_info(
                                        name, onnx_dtype, shape)
            for id, input_item in enumerate(self.graph.input):
                if input_item.name == name:
                    self.graph.input.remove(self.node_name2module[name])
                    self.node_name2module[name] = value_info
                    self.graph.input.append(value_info)
                    break
    def add_outputs(self, outputs):
        # https://github.com/onnx/onnx/issues/3277#issuecomment-1050600445
        output_names = outputs.values()
        accepted = []
        if len(output_names) == 0: return True
        ## sort nodes to  get_infered_shape correctly
        if self.need_topsort:
            self.toposort()
        inferred_value_info = get_infered_shape(self.model_proto)

        for info in inferred_value_info:
            if info.name in output_names:
                accepted.append(info.name)
                self.graph.output.append(info)
                self.graph_output_names.append("out_" + info.name)
                self.node_name2module["out_" + info.name] = info
        if set(output_names) != set(accepted):
            print("output:", list(set(output_names)-set(accepted))," not added! May caused by shape inference mismatch!")
            return False
        return True

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
                # more details about why the .flatten() is needed can be found in https://github.com/ZhangGe6/onnx-modifier/issues/28
                init_val_flat = init_val
                if len(init_val.shape) > 1:
                    init_val_flat = init_val.flatten()
                if len(init_val.shape) == 0:
                    init_val_flat = [init_val.item()]
                initializer_tensor = onnx.helper.make_tensor(
                    name=init_name,
                    data_type=np2onnxdtype(init_val.dtype),
                    dims=init_val.shape,
                    vals=init_val_flat)
                # print(initializer_tensor)
                self.initializer.append(initializer_tensor)
                self.initializer_name2module[init_name] = initializer_tensor
                # remove constant node replaced by initializer for some kind of node
                if init_name in self.cont_node_outname2name.keys():
                    self.graph.node.remove(self.node_name2module[self.cont_node_outname2name[init_name]])
                    del self.node_name2module[self.cont_node_outname2name[init_name]]
                    self.need_topsort = True

    def shape_inference(self):
        #add toposort to get correct infered shape
        if self.need_topsort:
            self.toposort()
        inferred_shape_info = get_infered_shape(copy.deepcopy(self.model_proto))

        del self.graph.value_info[:]
        output_info_bak = copy.deepcopy(self.graph.output)
        del self.graph.output[:]
        for info in inferred_shape_info:
            if "out_" + info.name in self.graph_output_names:
                self.graph.output.append(info)
            else:
                self.graph.value_info.append(info)
        # avoid to get an empty output list
        if len(self.graph.output) < 1:
            self.graph.output.extend(output_info_bak)

    def toposort(self):
        # inspired by graphsurgeon
        # https://github1s.com/NVIDIA/TensorRT/blob/master/tools/onnx-graphsurgeon/onnx_graphsurgeon/ir/graph.py
        def get_tensor2producer_map():
            tensor2producer_map = dict()
            for node in self.graph.node:
                for output in node.output:
                    tensor2producer_map[output] = node
            for inp in self.graph.input:
                tensor2producer_map[inp.name] = None

            return tensor2producer_map

        def get_input_nodes_map():
            input_nodes = dict()
            for node in self.graph.node:
                if node.name not in input_nodes.keys():
                    input_nodes[node.name] = []
                for inp in node.input:
                    # weights are not in tensor2producer_map
                    if inp in tensor2producer_map.keys():
                        producer = tensor2producer_map[inp]
                        input_nodes[node.name].append(producer)

            return input_nodes

        def get_hierarchy_level(node):
            if not node: return 0 # for input node
            if node.name in node_name2hierarchy:
                return node_name2hierarchy[node.name]

            # The level of a node is the level of it's highest input + 1.
            max_input_level = max([get_hierarchy_level(input_node) for input_node in input_nodes_map[node.name]] + [-1])

            return max_input_level + 1

        node_name2hierarchy = dict()
        tensor2producer_map = get_tensor2producer_map()
        input_nodes_map = get_input_nodes_map()
        for node in self.graph.node:
            node_name2hierarchy[node.name] = get_hierarchy_level(node)
        # print(node_name2hierarchy)

        sorted_node_names = [v[0] for v in sorted(node_name2hierarchy.items(), key=lambda x:x[1])]
        sorted_nodes = []
        for node_name in sorted_node_names:
            sorted_nodes.append(copy.deepcopy(self.node_name2module[node_name]))
        del self.graph.node[:]
        self.graph.node.extend(sorted_nodes)

    def post_process(self, kwargs):

        def get_tail_outputs():
            def collect_backtrack(input):
                if input not in input2nodes.keys(): # if the node has no child node
                    tail_outputs.add(input)
                    return

                node = input2nodes[input]
                if node in traversed_nodes: return  # if the node has been traversed
                traversed_nodes.append(node)

                for node in input2nodes[input]:
                    for output in node.output:
                        collect_backtrack(output)

            input2nodes = dict()
            for node in self.graph.node:
                for input in node.input:
                    if not (input in input2nodes.keys()):
                        input2nodes[input] = []
                    input2nodes[input].append(node)

            tail_outputs = set()
            traversed_nodes = []
            for inp in self.graph.input:
                collect_backtrack(inp.name)
            # print(tail_outputs)
            return tail_outputs

        def remove_isolated_nodes():
            def collect_reverse_backtrack(output):
                if output not in output2node.keys(): return # if the node has no parent node
                node = output2node[output]
                if node in connected_nodes: return # if the node has been traversed
                connected_nodes.append(node)

                for input in node.input:
                    collect_reverse_backtrack(input)

            output2node = dict()
            for node in self.graph.node:
                for output in node.output:
                    output2node[output] = node

            connected_nodes = []
            model_tail_outputs = get_tail_outputs()
            for output in model_tail_outputs:
                collect_reverse_backtrack(output)

            graph_connected_nodes = []
            graph_connected_initializers = []
            for node in self.graph.node:
                if node in connected_nodes:
                    graph_connected_nodes.append(copy.deepcopy(self.node_name2module[node.name]))
                    for inp in node.input:
                        if inp in self.initializer_name2module.keys():
                            graph_connected_initializers.append(copy.deepcopy(self.initializer_name2module[inp]))
            del self.graph.node[:]
            del self.initializer[:]
            self.graph.node.extend(graph_connected_nodes)
            self.initializer.extend(graph_connected_initializers)
            self.need_topsort = True

        useShapeInference = kwargs.pop("shapeInf", False)
        useCleanUp = kwargs.pop("cleanUp", False)

        if useShapeInference:
            self.shape_inference()
        if useCleanUp:
            print("[EXPERIMENTAL] Remove idle nodes...")
            remove_isolated_nodes()
        if self.need_topsort:
            self.toposort()

    def modify(self, modify_info):
        '''
        1. Some functions, such as modify_initializer(), should be placed
        before modify_node_io_name(), to avoid name mismatch error.
        2. add_nodes() should be placed at the first place, otherwise
        remove_node_by_node_states() will delete the initializer of
        newly added nodes by mistake.
        '''
        # print(modify_info['node_states'])
        # print(modify_info['node_renamed_io'])
        # print(modify_info['node_changed_attr'])
        # print(modify_info['added_node_info'])
        # print(modify_info['added_outputs'])
        #create a  return status to cope with output add failure
        status = True
        self.add_nodes(modify_info['added_node_info'], modify_info['node_states'])
        self.modify_initializer(modify_info['changed_initializer'])
        self.change_batch_size(modify_info['rebatch_info'])
        self.modify_node_io_name(modify_info['node_renamed_io'], modify_info['added_inputs'])
        self.remove_node_by_node_states(modify_info['node_states'])
        #added inputs here to avoid input remove cause by newly created input has the same name with deleted one
        self.add_inputs(modify_info['added_inputs'])
        #modify_inputs added after add_inputs to accept modification to newly added inputs
        self.modify_inputs(modify_info['modifed_inputs_info'])
        #mv add_outputs after modify_inputs to facilitate get_infered_shape which needs new input shape
        status = self.add_outputs(modify_info['added_outputs'])
        self.modify_node_attr(modify_info['node_changed_attr'])

        self.post_process(modify_info['postprocess_args'])
        return status

    def check_and_save_model(self, save_dir='./modified_onnx'):
        print("saving model...")
        import tkinter
        from tkinter import filedialog

        # onnx.checker.check_model(self.model_proto)
        if platform.system() == "Windows":
            window = tkinter.Tk()
            window.wm_attributes('-topmost', True)
            window.withdraw()
            save_path = filedialog.asksaveasfilename(
                parent=window,
                initialfile="modified_" + self.model_name[:-4]+'onnx',
                defaultextension=".onnx",
                filetypes=(("ONNX file", "*.onnx"),("All Files", "*.*"))
            )
        else:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, 'modified_' + self.model_name[:-4]+'onnx')

        if save_path:
            onnx.save(self.model_proto, save_path)
            print("model saved in {} !".format(save_dir))
            return save_path
        else:
            print("quit saving")
            return "NULLPATH"


    def check_and_save_json(self, save_dir='./modified_onnx'):
        print("saving json...")
        import tkinter
        from tkinter import filedialog
        if platform.system() == "Windows":
            window = tkinter.Tk()
            window.wm_attributes('-topmost', True)
            window.withdraw()
            save_path = filedialog.asksaveasfilename(
                parent=window,
                initialfile="modified_" + self.model_name[:-4]+'json',
                defaultextension=".json",
                filetypes=(("JSON file", "*.json"),("All Files", "*.*"))
            )
        else:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, 'modified_' + self.model_name[:-4]+'json')


        if save_path:
            onnx.checker.check_model(self.model_proto)
            message = MessageToJson(self.model_proto)
            with open(save_path, "w") as fo:
                fo.write(message)
            print("model saved in {} !".format(save_dir))
            return save_path
        else:
            print("quit saving")
            return "NULLPATH"

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
