# TODO
- [ ] beautify UI
- [ ] analyze and show model info on the index page, such as graph node number, model size etc.
- [ ] **ensure the model is fully loaded before modify() is called.**
    - otherwise `NameError: name 'onnx_modifier' is not defined` error will be invoked.
- [ ] support desktop application.
    - [x] Windows
        - [x] fix the shutdown issue
    - [ ] Linux
- [ ] support more flexible downloading schema
    - [ ] As this [request](https://github.com/ZhangGe6/onnx-modifier/pull/5) notes, the current downloading schema prevents `onnx-modifier ` from being deployed remotely as a service.
- [ ] support combine models (https://github.com/ZhangGe6/onnx-modifier/issues/63).
- [ ] support user-defined input/output number when the type of node's input/output is list.
- [x] refine the code.
    - [x] seperate the model graph rendering and editting into two classes in the js code for "cleaner" code.
    - [x] make the code more readable, cleaner and consistent in the code format.
    - [x] slim the code.
        - because some `.js` files (like electron.js and even python.js) in the `static` folder and `electron.html` in `templates` folder are legacies of Netron and can be further slimmed.
- [x] support adding model input/output node.
- [x] fix issue that "extra model inputs" emerges after deleting nodes. [issue#12](https://github.com/ZhangGe6/onnx-modifier/issues/12)
- [x] support adding more complicated nodes (which has some simple parameters like `reshape`).
- [x] add `shape inference` feature (mentioned in [this issue](https://github.com/ZhangGe6/onnx-modifier/issues/22))
- [x] set `cleanup` and `shape inference` as user-switch-on-off function, rather than automatic post-process.


# Some known reference issues/feature requests

- add node (experimentally supported)

    - (extend output's dim): https://github.com/onnx/onnx/issues/2709

    - add node (NMS): https://github.com/onnx/onnx/issues/2216

    - add node (add preprocess nodes): https://zhuanlan.zhihu.com/p/394395167

- combine models (not supported):

  - https://stackoverflow.com/questions/66178085/can-i-combine-two-onnx-graphs-together-passing-the-output-from-one-as-input-to

  - https://www.zhihu.com/people/kai-xin-zui-zhong-yao-76/posts

- modify attribute of nodes (supported)

    - topk: https://github.com/onnx/onnx/issues/2921
- remove layer (supported): https://github.com/onnx/onnx/issues/2638



