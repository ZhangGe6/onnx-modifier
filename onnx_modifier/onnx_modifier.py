# https://leimao.github.io/blog/ONNX-Python-API/
# https://leimao.github.io/blog/ONNX-IO-Stream/
# https://github.com/saurabh-shandilya/onnx-utils
# https://stackoverflow.com/questions/52402448/how-to-read-individual-layers-weight-bias-values-from-onnx-model

import os
import copy
import struct
import logging
import json
import numpy as np
import onnx
from onnx import numpy_helper
from .utils import str2np, str2val
from .utils import np2onnxdtype, str2onnxdtype
from .utils import make_new_node, make_attr_changed_node, make_input
from .utils import get_infered_shape

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
    def from_name_json_stream(cls, name, stream):
        from google.protobuf.json_format import Parse
        logging.info(f"loading model from json {name} ...")
        stream.seek(0)
        onnx_json = json.load(stream)
        onnx_str = json.dumps(onnx_json)
        model_proto = Parse(onnx_str, onnx.ModelProto())
        return cls(name, model_proto)

    @classmethod
    def from_name_protobuf_stream(cls, name, stream):
        # https://leimao.github.io/blog/ONNX-IO-Stream/
        logging.info("loading model...")
        stream.seek(0)
        model_proto = onnx.load_model(stream, "protobuf", load_external_data=False)
        logging.info("load done!")
        return cls(name, model_proto)

    def reload(self):
        self.model_proto = copy.deepcopy(self.model_proto_backup)
        self.graph = self.model_proto.graph
        self.initializer = self.model_proto.graph.initializer

        self.need_topsort = False
        self.gen_name2module_map()

    def gen_name2module_map(self):
        # node name => node
        self.node_name2module = dict()
        self.cst_node_outname2nodename = dict()
        for i, node in enumerate(self.graph.node):
            if node.name == '':
                node.name = str(node.op_type) + str(i)
            self.node_name2module[node.name] = node

            if node.op_type == "Constant":
                self.cst_node_outname2nodename[node.output[0]] = node.name

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

    def remove_node_by_node_states(self, node_states):
        # remove node from graph
        for node_name, node_state in node_states.items():
            if not (node_name in self.node_name2module.keys()):
                # for custom added node here
                continue
            if node_state == 'Deleted':
                if node_name in self.graph_output_names:
                    logging.debug('removing output {} ...'.format(node_name))
                    self.graph.output.remove(self.node_name2module[node_name])
                    self.graph_output_names = [n for n in self.graph_output_names if n != node_name]
                    self.node_name2module.pop(node_name, None)
                elif not node_name in self.graph_input_names:
                    logging.debug('removing node {} ...'.format(node_name))
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
            if input_name not in remained_inputs or \
               (input_name in node_states.keys() and node_states[input_name] == 'Deleted'):
                self.graph.input.remove(self.node_name2module[input_name])
                self.node_name2module.pop(input_name, None)

        self.need_topsort = True

    def change_node_io_name(self, node_renamed_io):
        # format of node_renamed_io : {node_name : {src_io_name : dst_io_name}}
        for node_name in node_renamed_io.keys():
            if node_name not in self.node_name2module.keys():
                # custom added nodes or custom added model outputs, or the deleted nodes
                continue

            renamed_ios = node_renamed_io[node_name]
            for src_name, dst_name in renamed_ios.items():
                node = self.node_name2module[node_name]
                if node_name in self.graph_input_names:
                    node.name = dst_name
                    self.graph_input_names.remove(src_name)
                    self.graph_input_names.append(dst_name)
                    self.node_name2module[dst_name] = node
                elif node_name in self.graph_output_names:
                    node.name = dst_name
                    self.graph_output_names.remove("out_" + src_name)
                    self.graph_output_names.append("out_" + dst_name)
                    self.node_name2module["out_" + dst_name] = node
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
                    # else rename output of the corresponding constant node and update cst_node_outname2nodename
                    elif src_name in self.cst_node_outname2nodename.keys():
                        cont = self.node_name2module[self.cst_node_outname2nodename[src_name]]
                        cont.output[0] = dst_name
                        self.cst_node_outname2nodename[dst_name] = self.cst_node_outname2nodename[src_name]
                        del self.cst_node_outname2nodename[src_name]

    def change_node_attr(self, node_changed_attr):
        # we achieve it by deleting the original node and make a (copied) new node
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
            node = make_new_node(node_info)
            self.graph.node.append(node)
            # update the node_name2module
            self.node_name2module[node.name] = node
        self.need_topsort = True

    def change_batch_size(self, rebatch_info):
        if not rebatch_info: return
        # https://github.com/onnx/onnx/issues/2182
        rebatch_type = rebatch_info['type']
        rebatch_value = rebatch_info['value']
        if rebatch_type == 'fixed':
            rebatch_value = int(rebatch_value)
            for tensor in self.graph.input:
                if type(rebatch_value) == str:
                    tensor.type.tensor_type.shape.dim[0].dim_param = rebatch_value
                elif type(rebatch_value) == int:
                    tensor.type.tensor_type.shape.dim[0].dim_value = rebatch_value
                else:
                    logging.warning('Unknown type {} for batch size. Fallback to dynamic batch size.'.format(type(rebatch_value)))
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

    def replace_primitive_input(self, new_input):
        self.graph.input.remove(self.node_name2module[new_input.name])
        self.graph.input.append(new_input)
        self.node_name2module[new_input.name] = new_input

    def add_new_input(self, new_input):
        self.graph_input_names.append(new_input.name)
        self.graph.input.append(new_input)
        self.node_name2module[new_input.name] = new_input

    def edit_inputs(self, inputs, rebatch_info):
        for input_info in inputs.values():
            inp = make_input(input_info)
            if inp.name in self.graph_input_names:
                logging.debug(f"Replacing the input {inp.name}")
                self.replace_primitive_input(inp)
            else:
                logging.debug(f"Adding new input {inp.name}")
                self.add_new_input(inp)

        if inputs:
            self.shape_inference()
        self.change_batch_size(rebatch_info)

    def add_outputs(self, outputs):
        # https://github.com/onnx/onnx/issues/3277#issuecomment-1050600445
        output_names = outputs.values()
        if len(output_names) == 0: return True
        # sort nodes to get_infered_shape correctly
        self.toposort()
        inferred_value_info = get_infered_shape(self.model_proto)
        inferred_name2value = {info.name : info for info in inferred_value_info}

        for name in output_names:
            if name in inferred_name2value.keys():
                info = inferred_name2value[name]
                self.graph.output.append(info)
                self.graph_output_names.append("out_" + info.name)
                self.node_name2module["out_" + info.name] = info
            else:
                logging.warning(f"{name} is not added successfully!")

    def change_initializer(self, changed_initializer):
        # print(changed_initializer)
        for init_name, meta in changed_initializer.items():
            # https://github.com/onnx/onnx/issues/2978
            init_type, init_val_str = meta
            if init_val_str == "": continue # in case we clear the input
            # print(init_name, init_type, init_val)
            init_val = str2np(init_val_str, init_type)
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
                if init_name in self.cst_node_outname2nodename.keys():
                    cst_node_name = self.cst_node_outname2nodename[init_name]
                    self.graph.node.remove(self.node_name2module[cst_node_name])
                    del self.node_name2module[cst_node_name]
                    self.need_topsort = True

    def shape_inference(self):
        self.toposort()
        inferred_shape_info = get_infered_shape(copy.deepcopy(self.model_proto))

        orig_output_info = copy.deepcopy(self.graph.output)
        orig_output_num = len(self.graph.output)
        del self.graph.value_info[:]
        del self.graph.output[:]

        for info in inferred_shape_info:
            if "out_" + info.name in self.graph_output_names:
                self.graph.output.append(info)
            else:
                self.graph.value_info.append(info)

        # recover the original ouptuts, to avoid output missing due to unperfect shape inference
        # TODO: this workaround can cause output shape mismatch if users change the input size
        if len(self.graph.output) < orig_output_num:
            self.graph.output.extend(orig_output_info)

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
        # TODO: check: does self.node_name2module still work?
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
            # NOTE(yancong): The initializer could be shared by multiple nodes. We should check
            # whether the initializer has been added to the initializer list before adding it.
            visited_initializer_names = set()
            for node in self.graph.node:
                if node in connected_nodes:
                    graph_connected_nodes.append(copy.deepcopy(self.node_name2module[node.name]))
                    for inp in node.input:
                        if inp in self.initializer_name2module.keys() and inp not in visited_initializer_names:
                            graph_connected_initializers.append(copy.deepcopy(self.initializer_name2module[inp]))
                            visited_initializer_names.add(inp)
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
            logging.warning("[EXPERIMENTAL] Remove idle nodes...")
            remove_isolated_nodes()
        if self.need_topsort:
            self.toposort()

    def modify(self, modify_info):
        '''
        The order of editing functions should be considered carefully
        '''
        logging.debug("=== modify_info ===\n", modify_info)

        self.add_nodes(modify_info['added_node_info'], modify_info['node_states'])
        self.change_initializer(modify_info['changed_initializer'])
        self.change_node_io_name(modify_info['node_renamed_io'])
        self.edit_inputs(modify_info['added_inputs'], modify_info['rebatch_info'])
        self.remove_node_by_node_states(modify_info['node_states'])
        self.add_outputs(modify_info['added_outputs'])
        self.change_node_attr(modify_info['node_changed_attr'])

        self.post_process(modify_info['postprocess_args'])

    def check_and_save_model(self, save_dir='./modified_onnx'):
        logging.info("saving model...")

        # onnx.checker.check_model(self.model_proto)
        save_dir = os.path.abspath(save_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, 'modified_' + self.model_name)

        if save_path:
            onnx.save(self.model_proto, save_path)
            logging.info("model saved in {} !".format(save_dir))
            return save_path
        else:
            return "NULL"

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
        outs = inference_session.run(output_names, {input_name: x})
        return outs
