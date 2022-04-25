# https://leimao.github.io/blog/ONNX-Python-API/
# https://github.com/saurabh-shandilya/onnx-utils

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
        for node in self.graph.node:
            if node.name == '':
                node.name = str(node.op_type) + str(node_idx)
                node_idx += 1

            self.node_name2module[node.name] = node
        
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

    def check_and_save_model(self, save_dir='./'):
        save_path = os.path.join(save_dir, 'modified_' + self.model_name)
        onnx.checker.check_model(self.model_proto)
        onnx.save(self.model_proto, save_path)
        
        
if __name__ == "__main__":
    model_path = "C:\\Users\\ZhangGe\\Desktop\\squeezenet1.0-12-int8.onnx"
    onnx_modifer = onnxModifier.from_model_path(model_path)
    onnx_modifer.remove_node_by_name('Softmax_nc_rename_64')
    onnx_modifer.check_and_save_model()
    
    
    

    
    