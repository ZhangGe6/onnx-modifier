# onnx-modifier update log

## 20230125
- fix the model output lost issue due to incomplete shape_inference
- refactor modify operations into modifier.js

## 20221209
- support removing legacy isolated nodes (like Constant) automatically
- support more flexible node attribute parsing
- some issue fixing and function improvements towards more user-friendly operations.
- add shape_inference feature when saving model

## 20221026
- support scrolling to the last page position when updating model
- support editing initializer feature

## 20220921
- add argparse module for custom config
- fix model parsing issue when the model is load in a drag-and-drop way
- support editing batch size

## 20220813

- support adding model input/output node. [issue 7](https://github.com/ZhangGe6/onnx-modifier/issues/7), [issue 8](https://github.com/ZhangGe6/onnx-modifier/issues/8), [issue 13](https://github.com/ZhangGe6/onnx-modifier/issues/13)
- fix issue that "extra model inputs" emerges after deleting nodes. [issue#12](https://github.com/ZhangGe6/onnx-modifier/issues/12)
- update windows executable file.

## 20220701

- support renaming model input/output

## 20220621
- add Dockerfile
  - thanks to [fengwang](https://github.com/fengwang) and [this PR](https://github.com/ZhangGe6/onnx-modifier/pulls?q=is%3Apr+is%3Aclosed)

## 20220620
- update graph automatically as soon as a modification is invoked.
- fix `shared arguments` issue.
- support editing the attributes and the name of inputs/outputs directly in the placeholders in the sidebar.
- newly support features:
    - Add new nodes.
    - edit the attributes of nodes.
- re-organize the layouts of buttons.
- add Windows executable file.

## 20220501
- the first public version. 
- support features:
    - delete/recover node.
    - rename the input and output of nodes.
- use sweetalert for nice alert.
- fix external data loading error for unix platform.

