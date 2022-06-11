# Add node

extend shape: https://github.com/onnx/onnx/issues/3920

add node (extend output's dim): https://github.com/onnx/onnx/issues/2709

add node （NMS）: https://github.com/onnx/onnx/issues/2216

add node (add preprocess nodes): https://zhuanlan.zhihu.com/p/394395167

# modify attribute of nodes

topk: https://github.com/onnx/onnx/issues/2921


# done

remove layer: https://github.com/onnx/onnx/issues/2638



# 或许可以帮助

http://yyixx.com/docs/algo/onnx/


# 待做的

bug(fixed): 不可连续添加某一种类型的节点（无反应）
boost: 直接使用侧边栏inputs/outputs属性框完成重命名，并提供reset功能
boost: 支持处理属性的修改
boost: 支持添加更复杂的节点
question: 在add()函数里，为什么对conv的inputs进行遍历，只能得到X，而得不到W和B？


# 其他
在修改节点输入输出时，建议修改方法是：把某一节点的输入，更改为另一节点的输出；而不是把某一节点的输出，改为另一节点的输入。