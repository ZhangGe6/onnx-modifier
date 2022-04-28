# https://leimao.github.io/blog/ONNX-Python-API/
# https://github.com/saurabh-shandilya/onnx-utils
# 
import io
import os
import onnx

class onnxModifier:
    def __init__(self, model_name, model_proto):
        self.model_name = model_name
        self.model_proto = model_proto
        self.graph = self.model_proto.graph
        
        self.gen_node_name2module_map()

    def gen_node_name2module_map(self):
        self.node_name2module = dict()
        node_idx = 0
        # for node in self.graph.input:
        #     node_idx += 1
            # self.node_name2module[node.name] = node
            
        for node in self.graph.node:
            if node.name == '':
                node.name = str(node.op_type) + str(node_idx)
            node_idx += 1
            self.node_name2module[node.name] = node
            
        for out in self.graph.output:
            self.node_name2module[out.name] = out
        self.graph_output_names = [out.name for out in self.graph.output]
        # print(self.node_name2module.keys())
        
    @classmethod
    def from_model_path(cls, model_path):
        model_name = os.path.basename(model_path)
        model_proto = onnx.load(model_path)
        return cls(model_name, model_proto)

    @classmethod
    def from_name_stream(cls, name, stream):
        # https://leimao.github.io/blog/ONNX-IO-Stream/
        stream.seek(0)
        model_proto = onnx.load_model(stream, onnx.ModelProto)
        return cls(name, model_proto)

    def remove_node_by_name(self, node_name):
        self.graph.node.remove(self.node_name2module[node_name])
    
    def remove_output_by_name(self, node_name):
        self.graph.output.remove(self.node_name2module[node_name])

    def remove_node_by_node_states(self, node_states):
        for node_name, node_state in node_states.items():
            if node_state == 'Deleted':
                if node_name in self.graph_output_names:
                    # print('removing output {} ...'.format(node_name))
                    self.remove_output_by_name(node_name)
                else:
                    # print('removing node {} ...'.format(node_name))
                    self.remove_node_by_name(node_name)

    def check_and_save_model(self, save_dir='./res_onnx'):
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
    onnx_modifier = onnxModifier.from_model_path(model_path)
    # for node in onnx_modifier.graph.node:
    #     print(node.name)
    # for node in onnx_modifier.graph.output:
    #     print(node.name)
    print(onnx_modifier.node_name2module.keys())
    print(onnx_modifier.graph_output_names)
    
    # onnx_modifier.remove_node_by_name('Softmax_nc_rename_64')
    # onnx_modifier.remove_output_by_name('softmaxout_1')
    # onnx_modifier.graph.output.remove(onnx_modifier.node_name2module['softmaxout_1'])
    # onnx_modifier.check_and_save_model()
    
    # print(type(onnx_modifier.graph.input))
    # print(type(onnx_modifier.graph.output))
    # print(onnx_modifier.graph.input)
    # print(onnx_modifier.graph.output)
    # print(onnx_modifier.node_name2module['Softmax_nc_rename_64'])
    # print(onnx_modifier.node_name2module['softmaxout_1'])
    # onnx_modifier.remove_node_by_name('softmaxout_1')
    
    
    # for node in onnx_modifier.graph.output:
    #     print(node.name)
        
    

    
    