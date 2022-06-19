# TODO

- support desktop application.

- support adding more complicated nodes (which has some simple parameters).

- support combine models.

- support user-defined input/output number when the type of node's input/output is list.

- slim the codes.
    - because some `.js` files (like electron.js and even python.js) in the `static` folder and `electron.html` in `templates` folder are legacy of Netron and can be further slimmed.


# Some known reference issues/feature requests

- add node (experimentally supported)

    - (extend output's dim): https://github.com/onnx/onnx/issues/2709

    - add node (NMS): https://github.com/onnx/onnx/issues/2216

    - add node (add preprocess nodes): https://zhuanlan.zhihu.com/p/394395167

combine models (not supported): 

    - https://stackoverflow.com/questions/66178085/can-i-combine-two-onnx-graphs-together-passing-the-output-from-one-as-input-to

    - https://www.zhihu.com/people/kai-xin-zui-zhong-yao-76/posts

- modify attribute of nodes (supported)

    - topk: https://github.com/onnx/onnx/issues/2921

- remove layer (supported): https://github.com/onnx/onnx/issues/2638




