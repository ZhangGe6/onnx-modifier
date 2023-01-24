var modifier = modifier || {};

modifier.Modifier = class {
    constructor(view) {
        this.view = view;
        this.model = null;
        this.graphs = null;
        this.name2ModelNode = new Map();
        this.name2ViewNode = new Map();
        this.name2NodeStates = new Map();
        this.addedOutputs = new Set();
        this.addedNode = new Map();

        this.namedEdges = new Map();
        this.changedAttributes = new Map();
        this.initializerEditInfo = new Map();
        this.renameMap = new Map();
        this.reBatchInfo = new Map();
        this._add_nodeKey = 0;
    }

    loadModelGraph(model, graphs) {
        this.model = model;
        this.graphs = graphs;
        this.graph = this.graphs[0];

        this.UpdateAddNodeDropDown();
    }

    // TODO: add filter feature like here: https://www.w3schools.com/howto/howto_js_dropdown.asp
    UpdateAddNodeDropDown() {
        // update dropdown supported node lost
        var addNodeDropdown = this.view._host.document.getElementById('add-node-dropdown');
        for (const node of this.model.supported_nodes) {
            // node: [domain, op]
            var option = new Option(node[1], node[0] + ':' + node[1]);
            // console.log(option)
            addNodeDropdown.appendChild(option);
        }
    }

    // ======= Record modified info =======> //
    addNode(op_domain, op_type) {
        var node_id = (this._add_nodeKey++).toString();  // in case input (onnx) node has no name
        var modelNodeName = 'custom_added_' + op_type + node_id;

        var properties = new Map();
        properties.set('domain', op_domain);
        properties.set('op_type', op_type);
        properties.set('name', modelNodeName);
        this.addedNode.set(modelNodeName, new view.LightNodeInfo(properties));

        this.applyAndUpdateView();
    }

    addModelOutput(node_name) {
        var modelNode = this.name2ModelNode.get(node_name);
        // use a output argument as a proxy
        this.addedOutputs.add(modelNode.outputs[0].arguments[0].name);
        this.applyAndUpdateView();
    }

    deleteSingleNode(node_name) {
        this.name2NodeStates.set(node_name, 'Deleted');
        this.name2ViewNode.get(node_name).element.style.opacity = 0.3;
    }

    deleteNodeWithChildren(node_name) {
        if (this.name2NodeStates.get(node_name) == 'Deleted') return;

        this.name2NodeStates.set(node_name, 'Deleted');
        this.name2ViewNode.get(node_name).element.style.opacity = 0.3;

        if (!this.namedEdges.has(node_name)) return; // for leaf node

        for (var i = 0; i < this.namedEdges.get(node_name).length; i++) {
            this.deleteNodeWithChildren(this.namedEdges.get(node_name)[i]);
        }
    }

    recoverSingleNode(node_name) {
        this.name2NodeStates.set(node_name, 'Exist');
        this.name2ViewNode.get(node_name).element.style.opacity = 1;
    }

    changeNodeInputOutput(modelNodeName, parameterName, param_type, param_index, arg_index, targetValue) {
        if (this.addedNode.has(modelNodeName)) {  // for custom added node 
            if (this.addedNode.get(modelNodeName).inputs.has(parameterName)) {
                var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];  // [arg.name, arg.is_optional]
                // update the corresponding initializer name
                if (this.initializerEditInfo.has(arg_name)) {
                    var init_val = this.initializerEditInfo.get(arg_name);
                    this.initializerEditInfo.set(targetValue, init_val);
                    this.initializerEditInfo.delete(arg_name);
                }
                this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0] = targetValue;
            }
            // console.log(this.initializerEditInfo)

            if (this.addedNode.get(modelNodeName).outputs.has(parameterName)) {
                this.addedNode.get(modelNodeName).outputs.get(parameterName)[arg_index][0] = targetValue;
            }
        }
        // console.log(this.addedNode)

        else {    // for the nodes in the original model
            var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
            // console.log(orig_arg_name)

            if (!this.renameMap.get(modelNodeName)) {
                this.renameMap.set(modelNodeName, new Map());
            }
            this.renameMap.get(modelNodeName).set(orig_arg_name, targetValue);
            // console.log(this.modifier.renameMap)
        }

        this.applyAndUpdateView();
    }

    getOriginalName(param_type, modelNodeName, param_index, arg_index) {
        if (param_type == 'model_input') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).arguments[0].original_name;
        }

        if (param_type == 'model_output') {
            // modelNodeName = 'out_' + modelNodeName
            // console.log(modelNodeName)
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).arguments[0].original_name;
            // console.log(orig_arg_name)
        }

        if (param_type == 'input') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).inputs[param_index].arguments[arg_index].original_name;
            // console.log(orig_arg_name)
        }
        if (param_type == 'output') {
            var orig_arg_name = this.name2ModelNode.get(modelNodeName).outputs[param_index].arguments[arg_index].original_name;
            // console.log(orig_arg_name)
        }

        return orig_arg_name;
    }

    changeNodeInputOutput(modelNodeName, parameterName, param_type, param_index, arg_index, targetValue) {
        if (this.addedNode.has(modelNodeName)) {  // for custom added node 
            if (this.addedNode.get(modelNodeName).inputs.has(parameterName)) {
                var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];  // [arg.name, arg.is_optional]
                // update the corresponding initializer name
                if (this.initializerEditInfo.has(arg_name)) {
                    var init_val = this.initializerEditInfo.get(arg_name);
                    this.initializerEditInfo.set(targetValue, init_val);
                    this.initializerEditInfo.delete(arg_name);
                }
                this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0] = targetValue;
            }
            // console.log(this.initializerEditInfo)

            if (this.addedNode.get(modelNodeName).outputs.has(parameterName)) {
                this.addedNode.get(modelNodeName).outputs.get(parameterName)[arg_index][0] = targetValue;
            }
        }
        // console.log(this.addedNode)

        else {    // for the nodes in the original model
            var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
            // console.log(orig_arg_name)

            if (!this.renameMap.get(modelNodeName)) {
                this.renameMap.set(modelNodeName, new Map());
            }
            this.renameMap.get(modelNodeName).set(orig_arg_name, targetValue);
            // console.log(this._renameMap)
        }
        // this.view._updateGraph()

        this.applyAndUpdateView();
    }

    changeInitializer(modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue) {
        var orig_arg_name = this.getOriginalName(param_type, modelNodeName, param_index, arg_index);
        this.initializerEditInfo.set(orig_arg_name, [type, targetValue]);
        // this.view._updateGraph()

        this.applyAndUpdateView();
    }

    changeAddedNodeInitializer(modelNodeName, parameterName, param_type, param_index, arg_index, type, targetValue) {
        var arg_name = this.addedNode.get(modelNodeName).inputs.get(parameterName)[arg_index][0];
        this.initializerEditInfo.set(arg_name, [type, targetValue]);
        // this.view._updateGraph()

        this.applyAndUpdateView();
    }

    changeNodeAttribute(modelNodeName, attributeName, targetValue, type) {
        if (this.addedNode.has(modelNodeName)) {
            this.addedNode.get(modelNodeName).attributes.set(attributeName, [targetValue, type]);
        }
        // console.log(this._addedNode)

        else {    // for the nodes in the original model
            if (!this.changedAttributes.get(modelNodeName)) {
                this.changedAttributes.set(modelNodeName, new Map());
            }
            this.changedAttributes.get(modelNodeName).set(attributeName, [targetValue, type]);

        }

        // this.view._updateGraph()
        this.applyAndUpdateView();
    }

    changeBatchSize(type, value) {
        if (type === "fixed") {
            this.reBatchInfo.set("type", "fixed");
            this.reBatchInfo.set("value", value);
        }
        else {  // dynamic
            this.reBatchInfo.set("type", "dynamic");
            this.reBatchInfo.set("value", "dynamic");
        }
    }
    // <======= Record modified info ======= //

    // ======= Apply modified info and update view =======> //
    deleteEnter() {
        this.applyAndUpdateView();
    }

    refreshModelInputOutput() {
        // console.log(this.modifier.renameMap)
        for (var input of this.graph._inputs) {
            var input_orig_name = input.arguments[0].original_name;
            if (this.renameMap.get(input_orig_name)) {
                var new_name = this.renameMap.get(input_orig_name).get(input_orig_name);
                var arg_with_new_name = this.graph._context.argument(new_name, input_orig_name);

                input.arguments[0] = arg_with_new_name;

                // change all the name of node input linked with model input meanwhile
                for (var node of this.graph._nodes) {
                    for (var node_input of node.inputs) {
                        for (const [index, element] of node_input.arguments.entries()) {
                            if (element.original_name == input_orig_name) {
                                var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                                node_input.arguments[index] = arg_with_new_name;

                                // save the changed name into _renameMap
                                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                                if (!this.renameMap.get(node.modelNodeName)) {
                                    this.renameMap.set(node.modelNodeName, new Map());
                                }

                                var orig_arg_name = element.original_name;
                                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
                            }
                        }
                    }
                }
            }
        }
        // console.log(this.graph.outputs)
        // create and add new output to graph
        this.graph.reset_custom_added_outputs();
        for (var output_name of this.addedOutputs) {
            this.graph.add_output(output_name);
        }
        // console.log(this.graph.outputs)
        for (var output of this.graph.outputs) {
            var output_orig_name = output.arguments[0].original_name;
            if (this.renameMap.get('out_' + output_orig_name)) {
                // for model input and output, node.modelNodeName == element.original_name
                var new_name = this.renameMap.get('out_' + output_orig_name).get(output_orig_name);
                // console.log(new_name)
                var arg_with_new_name = this.graph._context.argument(new_name, output_orig_name);

                output.arguments[0] = arg_with_new_name;

                // change all the name of node input linked with model input meanwhile
                for (var node of this.graph._nodes) {
                    for (var node_output of node.outputs) {
                        for (const [index, element] of node_output.arguments.entries()) {
                            if (element.original_name == output_orig_name) {
                                // console.log(element.original_name)
                                var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                                node_output.arguments[index] = arg_with_new_name;

                                // save the changed name into _renameMap
                                // as this modified _renamedMap, so refreshModelInputOutput() shoulf be called before refreshNodeArguments()
                                if (!this.renameMap.get(node.modelNodeName)) {
                                    this.renameMap.set(node.modelNodeName, new Map());
                                }

                                var orig_arg_name = element.original_name;
                                this.renameMap.get(node.modelNodeName).set(orig_arg_name, new_name);
                            }
                        }
                    }
                }
            }
        }
    }

    // re-generate the added node according to addedNode according to the latest addedNode
    refreshAddedNode() {
        this.graph.reset_custom_added_node();
        // for (const node_info of this.addedNode.values()) {
        // for (const [modelNodeName, node_info] of this.lastViewGraph.addedNode) {
        for (const [modelNodeName, node_info] of this.addedNode) {
            // console.log(node_info)
            var node = this.graph.make_custom_added_node(node_info);
            // console.log(node)

            for (const input of node.inputs) {
                var arg_list_info = [];
                for (const arg of input._arguments) {
                    arg_list_info.push([arg.name, arg.is_optional]);
                }
                this.addedNode.get(modelNodeName).inputs.set(input.name, arg_list_info);
            }

            for (const output of node.outputs) {
                var arg_list_info = [];
                for (const arg of output._arguments) {
                    arg_list_info.push([arg.name, arg.is_optional]);
                }
                this.addedNode.get(modelNodeName).outputs.set(output.name, arg_list_info);
            }

        }
    }

    // re-fresh node arguments in case the node inputs/outputs are changed
    refreshNodeArguments() {
        for (var node of this.graph._nodes) {
            // if (this.modifier.renameMap.get(node.modelNodeName)) {
            if (this.renameMap.get(node.modelNodeName)) {

                // check inputs
                for (var input of node.inputs) {
                    for (const [index, element] of input.arguments.entries()) {
                        if (this.renameMap.get(node.modelNodeName).get(element.original_name)) {
                            var new_name = this.renameMap.get(node.modelNodeName).get(element.original_name);
                            var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                            input.arguments[index] = arg_with_new_name;
                        }
                    }
                }

                // check outputs
                for (var output of node.outputs) {
                    for (const [index, element] of output.arguments.entries()) {
                        if (this.renameMap.get(node.modelNodeName).get(element.original_name)) {
                            var new_name = this.renameMap.get(node.modelNodeName).get(element.original_name);
                            // console.log(new_name)
                            var arg_with_new_name = this.graph._context.argument(new_name, element.original_name);

                            output.arguments[index] = arg_with_new_name;
                        }
                    }
                }
            }
        }
    }

    refreshNodeAttributes() {
        for (const node_name of this.changedAttributes.keys()) {
            var attr_change_map = this.changedAttributes.get(node_name);
            var node = this.name2ModelNode.get(node_name);

            for (var i = 0; i < node._attributes.length; ++i) {
                if (attr_change_map.get(node._attributes[i].name)) {
                    // [val, type]
                    node._attributes[i]._value = attr_change_map.get(node._attributes[i].name)[0];
                }
            }
        }
    }

    resetGraph() {
        // reset node states
        for (const name of this.name2NodeStates.keys()) {
            this.name2NodeStates.set(name, 'Exist');
        }

        // console.log(this.modifier.renameMap)
        // reset node inputs/outputs
        for (const changed_node_name of this.renameMap.keys()) {
            var node = this.name2ModelNode.get(changed_node_name);
            // console.log(node)
            // console.log(typeof node)
            // console.log(node.constructor.name)
            if (node.arguments) {   // model input or model output. Because they are purely onnx.Parameter
                // node.arguments[0] = this.graph._context.argument(node.modelNodeName);
                node.arguments[0] = this.graph._context.argument(node.arguments[0].original_name);
            }

            else {                   // model nodes
                //reset inputs
                for (var input of node.inputs) {
                    for (var i = 0; i < input.arguments.length; ++i) {
                        // console.log(input.arguments[i].original_name)
                        if (this.renameMap.get(node.modelNodeName).get(input.arguments[i].original_name)) {
                            input.arguments[i] = this.graph._context.argument(input.arguments[i].original_name);
                        }
                    }
                }

                // reset outputs
                for (var output of node.outputs) {
                    for (var i = 0; i < output.arguments.length; ++i) {
                        if (this.renameMap.get(node.modelNodeName).get(output.arguments[i].original_name)) {
                            output.arguments[i] = this.graph._context.argument(output.arguments[i].original_name);
                        }
                    }
                }

            }
        }
        this.renameMap = new Map();
        this.changedAttributes = new Map();
        this.reBatchInfo = new Map();
        this.initializerEditInfo = new Map();

        // clear custom added nodes
        this.addedNode = new Map();
        this.graph.reset_custom_added_node();
        this.addedOutputs = [];
        this.graph.reset_custom_added_outputs();

        // reset load location
        var container = this.view._getElementById('graph');
        container.scrollLeft = 0;
        container.scrollTop = 0;
        this.view._zoom = 1;

        this.applyAndUpdateView();
    }

    applyAndUpdateView() {
        this.refreshAddedNode();
        this.refreshModelInputOutput();
        this.refreshNodeArguments();
        this.refreshNodeAttributes();

        // this.graphs has been modified (inplace)
        this.view._updateGraph(this.model, this.graphs);
    }
    // <======= Apply modified info and update view ======= //

}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Modifier = modifier.Modifier;
}

