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

        this.namedEdges = new Map();

    }

    loadModelGraph(model, graphs) {
        this.model = model;
        this.graphs = graphs;
        this.graph = this.graphs[0];
    }

    addModelOutput(node_name) {
        var modelNode = this.name2ModelNode.get(node_name);
        // use a output argument as a proxy
        this.addedOutputs.add(modelNode.outputs[0].arguments[0].name);
        this.applyAndUpdateView();
    }

    // TODO: add _renameMap code
    refreshModelInputOutput() {
        this.graph.reset_custom_added_outputs();
        for (var output_name of this.addedOutputs) {
            this.graph.add_output(output_name);
        }
    }

    deleteSingleNode(node_name) {
        this.name2NodeStates.set(node_name, 'Deleted');
        this.name2ViewNode.get(node_name).element.style.opacity = 0.3;
    }

    recoverSingleNode(node_name) {
        this.name2NodeStates.set(node_name, 'Exist');
        this.name2ViewNode.get(node_name).element.style.opacity = 1;
    }

    deleteNodeWithChildren(node_name) {
        if (this.name2NodeStates.get(node_name)  == 'Deleted') 
        {
            return;
        }

        this.name2NodeStates.set(node_name, 'Deleted');
        this.name2ViewNode.get(node_name).element.style.opacity = 0.3;

        if (!this.namedEdges.has(node_name)){ // for leaf node
            return;
        }
        
        for (var i = 0; i < this.namedEdges.get(node_name).length; i++) {
            this.deleteNodeWithChildren(this.namedEdges.get(node_name)[i])
        }
    }

    deleteEnter() {
        this.applyAndUpdateView();
    }

    applyAndUpdateView() {
        this.refreshModelInputOutput();

        // this.graphs has been modified (inplace)
        this.view._updateGraph(this.model, this.graphs);


    }

}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Modifier = modifier.Modifier;
}

