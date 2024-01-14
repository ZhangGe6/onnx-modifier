# onnx-modifier update log

## 20231217
- Support editing initializer by reading from numpy file
- Support adding model input

## 20231123
- update onnx-metadata to support more advanced ops
- Fix onnx.load_model after onnx 1.15.0

## 20231030
- fix model inputs missing issue after deleting nodes that take them as input
- fix wrong backtrack when deleting nodes with changed outputs
- fix python package versions in requirements.txt to avoid version related issues

## 20230923
- Get modified batch value using shape inference, rather than hard-coded
- Add hyperlinks for the headers and reorder them
- support topological sort of nodes as a post process
- turn off onnx.checker.check_model by default
- Fix issue of adding output which are missing or with incorrect shape

## 20230819
- fix the issue of unsuccessful adding output that was deleted before
- fix issue of eddting scaler initializer

## 20230531
- fix adding node issue after resetting graph
- add requirements.txt to avoid python package incompatibility issue
- Use onnx-tool for more robust shape inference

## 20230312
- fix the issue that the newly added outputs are neglected
- support delete model outputs

## 20230222
- fix the parsing issue for scalar initilizer
- support more flexible numpy datatype parsing
- set shape inference and clean up as optional functions

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

