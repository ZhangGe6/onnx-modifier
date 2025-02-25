
var onnx = onnx || {};
var protobuf = protobuf || require('./protobuf');
var flatbuffers = flatbuffers || require('./flatbuffers');
var text = text || require('./text');

onnx.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (identifier.endsWith('saved_model.pb') || identifier.endsWith('predict_net.pb') || identifier.endsWith('init_net.pb')) {
            return undefined;
        }
        if (identifier.endsWith('predict_net.pbtxt') || identifier.endsWith('predict_net.prototxt') ||
            identifier.endsWith('init_net.pbtxt') || identifier.endsWith('init_net.prototxt')) {
            return undefined;
        }
        let tags = context.tags('pb');
        if (tags.size > 0) {
            if (tags.size === 1 && tags.get(1) === 2) {
                const tags = context.tags('pb+');
                const match = (tags, schema) => {
                    for (const pair of schema) {
                        const key = pair[0];
                        const inner = pair[1];
                        const value = tags[key];
                        if (value === undefined) {
                            continue;
                        }
                        if (inner === false) {
                            return false;
                        }
                        if (Array.isArray(inner)) {
                            if (typeof value !== 'object' || !match(value, inner)) {
                                return false;
                            }
                        }
                        else if (inner !== value) {
                            if (inner === 2 && !Array.isArray(value) && Object(value) === (value) && Object.keys(value).length === 0) {
                                return true;
                            }
                            return false;
                        }
                    }
                    return true;
                };
                // mediapipe.BoxDetectorIndex
                if (match(tags, [[1,[[1,[[1,[[1,5],[2,5],[3,5],[4,5],[6,0],[7,5],[8,5],[10,5],[11,0],[12,0]]],[2,5],[3,[]]]],[2,false],[3,false],[4,false],[5,false]]],[2,false],[3,false]] )) {
                    return undefined;
                }
                // third_party.tensorflow.python.keras.protobuf.SavedMetadata
                if (match(tags, [[1,[[1,[[1,0],[2,0]]],[2,0],[3,2],[4,2],[5,2]]]])) {
                    return undefined;
                }
            }
            if (Array.from(tags.keys()).every((tag) => tag <= 100) &&
                Array.from(tags.values()).every((type) => type < 5)) {
                // TensorProto
                if (tags.get(1) === 0 && tags.get(2) === 0) {
                    const schema = [[1,0],[2,0],[4,2],[5,2],[7,2],[8,2],[9,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                        return 'onnx.pb.TensorProto';
                    }
                }
                // GraphProto
                if (tags.get(1) === 2) {
                    const schema = [[1,2],[2,2],[3,2],[4,2],[5,2],[6,0],[7,0],[8,2],[9,2],[10,2],[11,2],[12,2],[13,2],[14,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                        const decode = (buffer, value) => {
                            const reader = protobuf.BinaryReader.open(buffer);
                            const length = reader.length;
                            while (reader.position < length) {
                                const tag = reader.uint32();
                                const number = tag >>> 3;
                                const type = tag & 7;
                                if (value === number) {
                                    return type === 2 ? reader.bytes() : null;
                                }
                                else {
                                    reader.skipType(type);
                                }
                            }
                            return null;
                        };
                        const stream = context.stream;
                        const buffer = stream.peek();
                        const nodeBuffer = decode(buffer, 1);
                        if (nodeBuffer) {
                            const nameBuffer = decode(nodeBuffer, 4);
                            if (nameBuffer && nameBuffer.every((c) => c > 0x20 && c < 0x7f)) {
                                return 'onnx.pb.GraphProto';
                            }
                        }
                    }
                }
                // ModelProto
                if (tags.get(7) === 2) {
                    const schema = [[1,0],[2,2],[3,2],[4,2][5,0],[6,2],[7,2],[8,2],[14,2],[20,2]];
                    if (schema.every((pair) => !tags.has(pair[0]) || tags.get(pair[0]) === pair[1])) {
                        return 'onnx.pb.ModelProto';
                    }
                }
            }
        }
        const stream = context.stream;
        if (stream.length > 5) {
            const buffer = stream.peek(Math.min(stream.length, 32));
            if (buffer[0] === 0x08 && buffer[1] < 0x0A && buffer[2] === 0x12) {
                const producers = [
                    'backend-test', 'BrainwaveCompiler',
                    'CNTK',
                    'keras2onnx', 'Kneron', 'kneron_formatter', 'kneron_kl530_test_case',
                    'darknet to ONNX example',
                    'htshinichi',
                    'MATLAB Deep Learning Toolbox Converter for ONNX Model Format', 'ML.NET', 'MVTec Software',
                    'onnx-caffe2', 'onnx-example', 'onnx.quantize', 'onnx.utils.extract_model', 'OnnxMLTools', 'onnx_test', 'onnxruntime-tools', 'onnxruntime.transformers',
                    'PaddlePaddle', 'pytorch',
                    'sclblonnx', 'skl2onnx',
                    'Tencent YouTu', 'tf2onnx', 'tflite2onnx',
                    'WinMLTools'
                ];
                if (producers.some((producer) => Array.from(producer).every((ch, index) => index + 4 < buffer.length && ch.charCodeAt(0) === buffer[index + 4]))) {
                    return 'onnx.pb.ModelProto';
                }
            }
        }
        if (onnx.Text.Reader.open(stream)) {
            return 'onnx.text';
        }
        if (onnx.JsonReader.open(context))
        {
            return 'onnx.json';
        }
        if (onnx.Runtime.Reader.open(stream, extension)) {
            return 'onnx.flatbuffers';
        }
        tags = context.tags('pbtxt');
        if (tags.has('ir_version')) {
            return 'onnx.pbtxt.ModelProto';
        }
        if (tags.has('graph') && extension !== 'model') {
            return 'onnx.pbtxt.ModelProto';
        }
        return undefined;
    }

    open(context, match) {
        const open = (model, format) => {
            return onnx.Metadata.open(context).then((metadata) => {
                return new onnx.Model(metadata, model, format);
            });
        };
        switch (match) {
            case 'onnx.pbtxt.ModelProto':
                return context.require('./onnx-proto').then(() => {
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.TextReader.open(stream);
                        const model = onnx.proto.ModelProto.decodeText(reader);
                        const format = 'ONNX' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File text format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.pb.TensorProto':
                return context.require('./onnx-proto').then(() => {
                    // TensorProto
                    // input_0.pb, output_0.pb
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.BinaryReader.open(stream);
                        const tensor = onnx.proto.TensorProto.decode(reader);
                        tensor.name = tensor.name || context.identifier;
                        const model = new onnx.proto.ModelProto();
                        model.graph = new onnx.proto.GraphProto();
                        model.graph.initializer = [ tensor ];
                        model.graph.value_info = [ new onnx.proto.ValueInfoProto() ];
                        model.graph.value_info[0].name = tensor.name;
                        model.graph.node = [ new onnx.proto.NodeProto() ];
                        model.graph.node[0].op_type = 'Constant';
                        model.graph.node[0].attribute = [ new onnx.proto.AttributeProto() ];
                        model.graph.node[0].attribute[0].name = 'value';
                        model.graph.node[0].attribute[0].type = onnx.AttributeType.TENSOR;
                        model.graph.node[0].attribute[0].t = tensor;
                        const format = 'ONNX Tensor';
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.TensorProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.pb.GraphProto':
                return context.require('./onnx-proto').then(() => {
                    // GraphProto
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.BinaryReader.open(stream);
                        const model = new onnx.proto.ModelProto();
                        model.graph = onnx.proto.GraphProto.decode(reader);
                        const format = 'ONNX';
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.GraphProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.pb.ModelProto':
                return context.require('./onnx-proto').then(() => {
                    // ModelProto
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = protobuf.BinaryReader.open(stream);
                        const model = onnx.proto.ModelProto.decode(reader);
                        const format = 'ONNX' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        // console.log(format)  // ONNX v7
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            case 'onnx.flatbuffers': {
                return context.require('./onnx-schema').then((/* schema */) => {
                    try {
                        onnx.schema = flatbuffers.get('ort').onnxruntime.fbs;
                        const stream = context.stream;
                        const reader = onnx.Runtime.Reader.open(stream, 'ort');
                        const model = reader.read();
                        const format = 'ONNX Runtime' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not ort.Model (' + message.replace(/\.$/, '') + ').');
                    }
                });
            }
            case 'onnx.text': {
                return context.require('./onnx-proto').then(() => {
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const stream = context.stream;
                        const reader = onnx.Text.Reader.open(stream);
                        const model = reader.read();
                        const format = 'ONNX Text' + (model.ir_version ? ' v' + model.ir_version.toString() : '');
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            }
            case 'onnx.json': {
                return context.require('./onnx-proto').then(async () => {
                    try {
                        onnx.proto = protobuf.get('onnx').onnx;
                        const reader = onnx.JsonReader.open(context);
                        await reader.read();
                        const model = reader.model;
                        const format = reader.format;
                        return open(model, format);
                    }
                    catch (error) {
                        const message = error && error.message ? error.message : error.toString();
                        throw new onnx.Error('File format is not onnx.ModelProto (' + message.replace(/\.$/, '') + ').');
                    }
                });
            }
            default: {
                throw new onnx.Error("Unknown ONNX format '" + match + "'.");
            }
        }
    }
};

onnx.Model = class {

    constructor(metadata, model, format) {
        this._graphs = [];
        this._format = format;
        this._producer = model.producer_name && model.producer_name.length > 0 ? model.producer_name + (model.producer_version && model.producer_version.length > 0 ? ' ' + model.producer_version : '') : null;
        this._domain = model.domain;
        this._modelVersion = model.model_version;
        this._description = model.doc_string;
        this._metadata = [];
        this._imports = null;

        const imports = new Map();
        if (model.opset_import && model.opset_import.length > 0) {
            for (const opset_import of model.opset_import) {
                const domain = opset_import.domain || 'ai.onnx';
                const version = opset_import.version ? typeof opset_import.version === 'number' ? opset_import.version: opset_import.version.toNumber() : 0;
                if (!imports.has(domain) || imports.get(domain) > version) {
                    imports.set(domain, version);
                }
            }
            this._imports = Array.from(imports).map((pair) => pair[0] + ' v' + pair[1].toString());
        }
        if (imports.size == 0) {
            imports.set('ai.onnx', 1);
            imports.set('ai.onnx.ml', 1);
        }

        let imageFormat = '';
        if (model.metadata_props) {
            const imageMetadata = {};
            for (const metadata_prop of model.metadata_props) {
                switch (metadata_prop.key) {
                    case 'author':
                        this._author = metadata_prop.value;
                        break;
                    case 'company':
                        this._company = metadata_prop.value;
                        break;
                    case 'converted_from':
                        this._converted_from = metadata_prop.value;
                        break;
                    case 'license':
                        this._license = metadata_prop.value;
                        break;
                    case 'license_url':
                        this._licenseUrl = metadata_prop.value;
                        break;
                    case 'Image.BitmapPixelFormat':
                    case 'Image.ColorSpaceGamma':
                    case 'Image.NominalPixelRange':
                        imageMetadata[metadata_prop.key] = metadata_prop.value;
                        break;
                    default:
                        this._metadata.push({ name: metadata_prop.key, value: metadata_prop.value});
                        break;
                }
            }
            imageFormat = [ imageMetadata['Image.BitmapPixelFormat'], imageMetadata['Image.ColorSpaceGamma'], imageMetadata['Image.NominalPixelRange'] ].filter((item) => item);
        }
        this._graphs = [];
        if (model && model.graph) {
            // const graphMetadata = new onnx.GraphMetadata(metadata, imports);
            // const context = new onnx.ModelContext(graphMetadata, imageFormat);
            this.graphMetadata = new onnx.GraphMetadata(metadata, imports);
            const context = new onnx.ModelContext(this.graphMetadata, imageFormat);
            for (const func of model.functions || []) {
                context.metadata.add(new onnx.Function(context, func));
            }
            // var tmp = this.supported_nodes
            const graphs = [ model.graph ];
            while (graphs.length > 0) {
                const graph = graphs.shift();
                this._graphs.push(context.graph(graph));
                for (const node of graph.node || []) {
                    for (const attribute of node.attribute || []) {
                        if (attribute.g) {
                            graphs.push(attribute.g);
                        }
                        else if (attribute.graphs && attribute.graphs.length > 0) {
                            graphs.push(...attribute.graphs);
                        }
                    }
                }
            }
        }
    }

    get format() {
        return this._format;
    }

    get imports() {
        return this._imports;
    }

    get producer() {
        return this._producer;
    }

    get domain() {
        return this._domain || null;
    }

    get description() {
        return this._description || null;
    }

    get author() {
        return this._author || null;
    }

    get company() {
        return this._company || null;
    }

    get source() {
        return this._converted_from || null;
    }

    get license() {
        const license = [];
        if (this._license && this._license.length > 0) {
            license.push(this._license);
        }
        if (this._licenseUrl && this._licenseUrl.length > 0) {
            license.push('<a href=\'' + this._licenseUrl + '\'>' + this._licenseUrl + '</a>');
        }
        if (license.length > 0) {
            return license;
        }
        return null;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }

    get supported_nodes() {
        // console.log(this.graphMetadata);
        var nodes = []
        for (const domain of this.graphMetadata._metadata._map.keys()) {
            // console.log(domain)
            for (const op of this.graphMetadata._metadata._map.get(domain).keys()) {
                // console.log(op)
                nodes.push([domain, op])
            }
        }
        return nodes
    }

};

onnx.Graph = class {
    // context is ModelContext here
    constructor(context, graph) {
        this._node = '';
        this._description = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._name = graph.name || null;
        this._description = graph.doc_string || '';
        this._value_info = graph.value_info || null;
        this._graph = graph;

        context = new onnx.GraphContext(context, graph.node);
        this._context = context;

        this._custom_add_node_io_idx = 0
        this._custom_added_node = []
        this._custom_added_outputs = []
        this._custom_deleted_outputs = []
        this._custom_added_inputs = []
        this._custom_deleted_inputs = []

        // model parameter assignment here!
        // console.log(graph)
        for (const initializer of graph.initializer) {
            const tensor = context.tensor(initializer.name);
            // console.log(initializer)   // type: TensorProto
            tensor.initializer = new onnx.Tensor(context, initializer, 'Initializer');
        }
        for (const sparse_initializer of graph.sparse_initializer) {
            const tensor = context.tensor(sparse_initializer.values.name);
            tensor.initializer = new onnx.Tensor(context, sparse_initializer, 'Sparse Initializer');
        }
        for (const tensor_annotation of graph.quantization_annotation || []) {
            const tensor = context.tensor(tensor_annotation.tensor_name);
            const annotation = {};
            for (const pair of tensor_annotation.quant_parameter_tensor_names) {
                annotation[pair.key] = pair.value;
            }
            tensor.annotation = annotation;
        }
        for (const valueInfo of graph.value_info) {
            const tensor = context.tensor(valueInfo.name);
            tensor.type = context.createType(valueInfo.type);
            tensor.description = valueInfo.doc_string;
        }
        graph.input = graph.input.map((valueInfo) => {
            const tensor = context.tensor(valueInfo.name);
            tensor.type = context.createType(valueInfo.type);
            tensor.description = valueInfo.doc_string;
            return tensor;
        });
        graph.output = graph.output.map((valueInfo) => {
            const tensor = context.tensor(valueInfo.name);
            tensor.type = context.createType(valueInfo.type);
            tensor.description = valueInfo.doc_string;
            return tensor;
        });
        new onnx.Inference(graph.node, graph.output);
        context.push(graph.node, graph.input, graph.output);
        this._nodes = context.pop();    // get context._nodes() #Line1727
        for (const input of graph.input) {
            const argument = context.argument(input.name);
            if (!argument.initializer) {
                this._inputs.push(new onnx.Parameter(input.name, [ argument ]));
            }
        }
        for (const output of graph.output) {
            const argument = context.argument(output.name);
            if (!argument.initializer) {
                this._outputs.push(new onnx.Parameter(output.name, [ argument ]));
            }
        }
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
    }

    get inputs() {
        // return this._inputs;
        var filtered_inputs = [];
        // when the original model input shape is changed (onnxModifier.changeModelInput())
        var _custom_added_inputs_names = [];
        for (const inp of this._custom_added_inputs) {
            _custom_added_inputs_names.push(inp.name);
        }
        for (const inp of this._inputs) {
            if (_custom_added_inputs_names.includes(inp.name)) continue;
            filtered_inputs.push(inp);
        }

        var all_inputs = filtered_inputs.concat(this._custom_added_inputs);
        filtered_inputs = [];
        for (const inp of all_inputs) {
            if (this._custom_deleted_inputs.includes(inp.name)) continue;
            filtered_inputs.push(inp);
        }
        return filtered_inputs;
    }

    get outputs() {
        // return this._outputs;
        var all_outputs = this._outputs.concat(this._custom_added_outputs);
        var filtered_outputs = [];
        for (const out of all_outputs) {
            if (this._custom_deleted_outputs.includes(out.name)) continue;
            filtered_outputs.push(out);
        }
        return filtered_outputs;
    }

    get nodes() {
        // return this._nodes;
        return this._nodes.concat(this._custom_added_node);
    }

    reset_custom_added_node() {
        this._custom_added_node = []
        // this._custom_add_node_io_idx = 0
    }

    toString() {
        return 'graph(' + this.name + ')';
    }

    make_custom_added_node(node_info) {
        // type of node_info == LightNodeInfo
        const schema = this._context.metadata.type(node_info.properties.get('op_type'), node_info.properties.get('domain'));
        // console.log(schema)

        // console.log(node_info.attributes)
        // console.log(node_info.inputs)
        // console.log(node_info.outputs)
        // var max_input = schema.max_input
        // var min_input = schema.max_input
        var max_custom_add_input_num = Math.min(schema.max_input, 8)  // set at most 8 custom_add inputs
        var max_custom_add_output_num = Math.min(schema.max_output, 8)  // set at most 8 custom_add outputs

        // console.log(node_info)
        var inputs = []
        // use "if" to deal with node type with no input, like Constant
        if (schema.inputs) {
            for (let i = 0; i < schema.inputs.length; ++i) {
                const input = schema.inputs[i]

                var node_info_input = node_info.inputs.get(input.name)
                // console.log(node_info_input)

                var arg_list = []
                if (input.list) {
                    for (let j = 0; j < max_custom_add_input_num; ++j) {
                        if (node_info_input && node_info_input[j]) {
                            var arg_name = node_info_input[j][0]  // [arg.name, arg.is_optional]
                        }
                        else {
                            var arg_name = 'list_custom_input_' + (this._custom_add_node_io_idx++).toString()
                        }
                        arg_list.push(this._context.argument(arg_name))
                    }
                }
                else {
                    if (node_info_input && node_info_input[0]) {
                        var arg_name = node_info_input[0][0]  // [arg.name, arg.is_optional]
                    }
                    else {
                        var arg_name = 'custom_input_' + (this._custom_add_node_io_idx++).toString()
                    }
                    arg_list = [this._context.argument(arg_name)]
                }

                for (var arg of arg_list) {
                    arg.is_custom_added = true;
                    if (input.option && input.option == 'optional') {
                        arg.is_optional = true;
                    }
                }
                inputs.push(new onnx.Parameter(input.name, arg_list));
            }
        }

        var outputs = []
        if (schema.outputs) {
            for (let i = 0; i < schema.outputs.length; ++i) {
                const output = schema.outputs[i]
                var node_info_output = node_info.outputs.get(output.name)

                var arg_list = []
                if (output.list) {
                    for (let j = 0; j < max_custom_add_output_num; ++j) {
                        if (node_info_output && node_info_output[j]) {
                            var arg_name = node_info_output[j][0]
                        }
                        else {
                            var arg_name = 'list_custom_output_' + (this._custom_add_node_io_idx++).toString()
                        }
                        arg_list.push(this._context.argument(arg_name))
                    }
                }
                else {
                    if (node_info_output && node_info_output[0]) {
                        var arg_name = node_info_output[0][0]
                    }
                    else {
                        var arg_name = 'custom_output_' + (this._custom_add_node_io_idx++).toString()
                    }

                    arg_list = [this._context.argument(arg_name)]
                }

                for (var arg of arg_list) {
                    arg.is_custom_added = true;
                    if (output.option && output.option == 'optional') {
                        arg.is_optional = true;
                    }
                }
                outputs.push(new onnx.Parameter(output.name, arg_list));
            }
        }

        // console.log(inputs)
        // console.log(outputs)

        // console.log(node_info)
        var attributes = []
        if (schema.attributes) {
            for (const attr of schema.attributes) {
                // [value, type]
                // console.log(node_info.attributes)
                var value = null;
                if (node_info.attributes.has(attr.name)) {
                    value = node_info.attributes.get(attr.name)[0];
                }
                attributes.push(
                    new onnx.LightAttributeInfo(
                        attr.name,
                        attr.description,
                        attr.type,
                        value
                        )
                    )
            }
        }
        // console.log(attributes)
        var custom_add_node = new onnx.Node(
                this._context,
                node_info.properties.get('op_type'),
                node_info.properties.get('domain'),
                node_info.properties.get('name'),
                null, // schema.description, // omit it to save sidebar space. The node description can also be seen in the node `type` expander
                attributes,
                inputs,
                outputs
            );
        // console.log(custom_add_node)

        this._custom_added_node.push(custom_add_node)

        return custom_add_node;
    }

    reset_custom_modified_outputs() {
        this._custom_added_outputs = [];
        this._custom_deleted_outputs = [];
    }

    add_output(name) {
        const argument = this._context.argument(name);
        this._custom_added_outputs.push(new onnx.Parameter(name, [ argument ]));
    }

    delete_output(name) {
        this._custom_deleted_outputs.push(name);
    }

    reset_custom_modified_inputs() {
        this._custom_added_inputs = [];
        this._custom_deleted_inputs = [];
    }

    add_input(name_shape_type) {
        // [name, type[shape]]
        var name = name_shape_type[0];
        var shape_type = name_shape_type[1];
        // valid shape format: dtype[shape], like float32[1,3,224,224]
        var dtype = shape_type.split("[")[0];
        var shape_str = shape_type.split("[")[1].split("]")[0];

        var shape = [];
        for (const dim of shape_str.split(",")) {
            shape.push(parseInt(dim));
        }
        // console.log(dtype, shape);

        const tensor = this._context.tensor(name);
        tensor.type = new onnx.TensorType(dtype, new onnx.TensorShape(shape));

        const argument = this._context.argument(name);
        this._custom_added_inputs.push(new onnx.Parameter(name, [ argument ]));
        // console.log(this._custom_added_inputs);

        // remove the input from the deleted list if it was deleted before
        var deleted_inputs = [];
        for (var input in this._custom_deleted_inputs) {
            if (input.name != name) {
                deleted_inputs.push(input);
            }
        }
        this._custom_deleted_inputs = deleted_inputs;
    }

    delete_input(name) {
        this._custom_deleted_inputs.push(name);
        // this._graph.value_info = [];
    }
};

onnx.Parameter = class {

    constructor(name, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }
};

onnx.Argument = class {

    constructor(name, type, initializer, annotation, description, original_name) {
        if (typeof name !== 'string') {
            throw new onnx.Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._annotation = annotation;
        this._description = description || '';

        this.original_name = original_name || name;
        this.is_custom_added = false;
        this.is_optional = false;

    }

    get name() {
        return this._name;
    }

    // https://bobbyhadz.com/blog/javascript-cannot-set-property-which-has-only-getter
    // It is unsafe
    set name(name) {
        this._name = name;
    }

    get type() {
        return this._type;
    }

    set type(type) {
        this._type = type;
    }

    get description() {
        return this._description;
    }

    get quantization() {
        if (this._annotation) {
            return Object.keys(this._annotation).map((key) => key + ': ' + this._annotation[key]).join(', ');
        }
        return null;
    }

    get initializer() {
        return this._initializer;
    }
};

onnx.Node = class {

    constructor(context, op_type, domain, name, description, attributes, inputs, outputs) {
        attributes = attributes || [];
        this._type = context.metadata.type(op_type, domain) || { name: op_type, module: domain };
        if (this.type.module !== domain && !(this._type instanceof onnx.Function)) {
            this._type = Object.assign({}, this.type);
            this._type.name = op_type;
            this._type.module = domain;
        }
        this._name = name || '';
        this._description = description || '';
        this._inputs = inputs;
        this._outputs = outputs;
        // console.log(attributes)
        this._attributes = attributes.map((attribute) => new onnx.Attribute(context, op_type, domain, attribute));
        // console.log(this._attributes)
        this._chain = [];
        const identifier = domain ? domain + '.' + op_type : op_type;
        switch (identifier) {
            case 'com.microsoft.FusedConv': {
                const activation = attributes.find((attribute) => attribute.name === 'activation');
                if (activation) {
                    const type = context.decodeText(activation.s);
                    this._chain.push(new onnx.Node(context, type, '', '', '', [], [], []));
                }
                break;
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get chain() {
        return this._chain;
    }
};

onnx.Attribute = class {
    // `context` here is GraphContext
    constructor(context, op_type, domain, attribute) {
        this._name = attribute.name;
        this._description = attribute.doc_string || attribute.description || '';
        this._type = null;
        this._value = null;

        switch (attribute.type) {
            case onnx.AttributeType.FLOAT:
                this._value = attribute.f;
                this._type = 'float32';
                break;
            case onnx.AttributeType.INT:
                this._value = attribute.i;
                this._type = 'int64';
                break;
            case onnx.AttributeType.STRING:
                switch (op_type) {
                    case 'Int8GivenTensorFill':
                        this._value = Array.from(attribute.s);
                        break;
                    default:
                        this._value = context.decodeText(attribute.s);
                        break;
                }
                this._type = 'string';
                break;
            case onnx.AttributeType.TENSOR:
                this._value = new onnx.Tensor(context, attribute.t);
                this._type = 'tensor';
                break;
            case onnx.AttributeType.GRAPH:
                this._value = context.graph(attribute.g);
                this._type = 'graph';
                break;
            case onnx.AttributeType.FLOATS:
                this._value = ArrayBuffer.isView(attribute.floats) ? Array.from(attribute.floats) : attribute.floats;
                this._type = 'float32[]';
                break;
            case onnx.AttributeType.INTS:
                this._value = ArrayBuffer.isView(attribute.ints) ? Array.from(attribute.ints) : attribute.ints;
                this._type = 'int64[]';
                break;
            case onnx.AttributeType.STRINGS:
                this._value = attribute.strings.map((s) => context.decodeText(s));
                this._type = 'string[]';
                break;
            case onnx.AttributeType.TENSORS:
                this._value = attribute.tensors.map((tensor) => new onnx.Tensor(context, tensor));
                this._type = 'tensor[]';
                break;
            case onnx.AttributeType.GRAPHS:
                this._value = attribute.graphs.map((graph) => context.graph(graph));
                this._type = 'graph[]';
                break;
            case onnx.AttributeType.SPARSE_TENSOR:
                this._value = new onnx.Tensor(context, attribute.sparse_tensor);
                this._type = 'tensor';
                break;
            case onnx.AttributeType.SPARSE_TENSORS:
                this._value = attribute.sparse_tensors.map((tensor) => new onnx.Tensor(context, tensor));
                this._type = 'tensor[]';
                break;
            case onnx.AttributeType.TYPE_PROTO:
                this._value = context.createType(attribute.tp);
                this._type = 'type';
                break;
            case onnx.AttributeType.TYPE_PROTOS:
                this._value = attribute.type_protos.map((type) => context.createType(type));
                this._type = 'type[]';
                break;
            default:
                // console.log(attribute)
                this._value = attribute.value;
                this._type = attribute.type;
                // TODO: I comment the Error message for the compatibility of onnx.Graph.make_custom_added_node. This is may be unsafe
                // throw new onnx.Error("Unknown attribute type '" + attribute.type + "'.");
        }

        // see #L1294 GraphMetadata
        const metadata = context.metadata.attribute(op_type, domain, attribute.name);
        // console.log(metadata)
        if (metadata) {
            if (Object.prototype.hasOwnProperty.call(metadata, 'default') && this._value == metadata.default) {
                this._visible = false;
            }
            if (metadata.type === 'DataType') {
                this._type = metadata.type;
                const value = this._value ? parseInt(this._value.toString(), 10) : this._value;
                // this._value = Number.isInteger(value) ? context.createDataType(value) : value;
                if (value != NaN && Number.isInteger(value)) {
                    this._value = context.createDataType(value);
                }
                // console.log(attribute.type, attribute.value)
                // console.log(this._type, value, this._value)
            }
        }
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get value() {
        return this._value;
    }

    get description() {
        return this._description;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

onnx.LightAttributeInfo = class {
    constructor(name, description, type, value) {
        this.name = name;
        this.description = description;
        this.type = type;
        // this.value = value || null;
        // console.log(value, value || null) // TODO: amazing output: 0, null
        this.value = value
    }
}

onnx.Group = class {

    constructor(name, groups) {
        this._type = { name: 'Scope' };
        this._name = name;
        this._nodes = [];
        for (const entry of groups) {
            const key = entry[0];
            if (key === '') {
                for (const node of entry[1]) {
                    this._nodes.push(node);
                }
            }
            else {
                this._nodes.push(new onnx.Group(name === '' ? key : name + '/' + key, entry[1]));
            }
        }
        const set = new Set();
        const inputs = new Array();
        const outputs = new Array();
        for (const node of this._nodes) {
            if (node instanceof onnx.Group) {
                node.freeze();
            }
            for (const parameter of node.outputs) {
                for (const argument of parameter.arguments) {
                    if (!argument.initializer) {
                        outputs.push(argument);
                        set.add(argument.name);
                    }
                }
            }
        }
        for (const node of this._nodes) {
            for (const parameter of node.inputs) {
                for (const argument of parameter.arguments) {
                    if (!set.has(argument.name) && !argument.initializer) {
                        inputs.push(argument);
                    }
                }
            }
        }
        this._inputs = [ new onnx.Parameter('inputs', inputs) ];
        this._outputs = [ new onnx.Parameter('outputs', outputs) ];
        this._attributes = [];
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

    get nodes() {
        return this._nodes;
    }
};

onnx.Tensor = class {

    constructor(context, tensor, kind) {
        this._kind = kind || null;
        const data = (tensor) => {
            let data = undefined;
            if (tensor.data_location === onnx.DataLocation.DEFAULT) {
                switch (tensor.data_type) {
                    case onnx.DataType.FLOAT16:
                        if (tensor.int32_data && tensor.int32_data.length > 0) {
                            const buffer = new Uint8Array(tensor.int32_data.length << 1);
                            const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                            const array = tensor.int32_data;
                            for (let i = 0; i < array.length; i++) {
                                view.setUint16(i << 1, array[i], true);
                            }
                            data = {
                                type: tensor.data_type,
                                buffer: buffer
                            };
                        }
                        break;
                    case onnx.DataType.FLOAT:
                        data = new Float32Array(tensor.float_data);
                        break;
                    case onnx.DataType.DOUBLE:
                        data = new Float64Array(tensor.double_data);
                        break;
                    case onnx.DataType.BOOL:
                        if (tensor.int32_data && tensor.int32_data.length > 0) {
                            const array = tensor.int32_data;
                            data = new Array(array.length);
                            for (let i = 0; i < data.length; i++) {
                                data[i] = array[i] === 0 ? false : true;
                            }
                        }
                        break;
                    case onnx.DataType.INT8:
                        data = new Int8Array(tensor.int32_data);
                        break;
                    case onnx.DataType.UINT8:
                        data = new Uint8Array(tensor.int32_data);
                        break;
                    case onnx.DataType.INT16:
                        data = new Int32Array(tensor.int32_data);
                        break;
                    case onnx.DataType.UINT16:
                        data = new Int32Array(tensor.int32_data);
                        break;
                    case onnx.DataType.INT32:
                        data = new Int32Array(tensor.int32_data);
                        break;
                    case onnx.DataType.UINT32:
                    case onnx.DataType.UINT64:
                        data = tensor.uint64_data;
                        break;
                    case onnx.DataType.INT64:
                        data = tensor.int64_data;
                        break;
                    case onnx.DataType.STRING:
                        data = tensor.string_data;
                        break;
                }
                if (data && (Array.isArray(data) || ArrayBuffer.isView(data)) && data.length === 0) {
                    data = undefined;
                }
                if (!data && tensor.raw_data && tensor.raw_data.length > 0) {
                    data = {
                        type: tensor.data_type,
                        buffer: tensor.raw_data
                    };
                }
            }
            return data;
        };
        if ((onnx.proto && tensor instanceof onnx.proto.SparseTensorProto) ||
            (onnx.schema && tensor instanceof onnx.schema.SparseTensor)) {
            this._name = tensor.values.name || '';
            this._type = context.createTensorType(tensor.values.data_type, tensor.dims.map((dim) => dim), null);
            this._location = Array.from(new Set([ context.createLocation(tensor.values.data_location), context.createLocation(tensor.indices.data_location) ])).join(':');
            this._values = data(tensor.values);
            this._indices = data(tensor.indices);
        }
        else {
            this._name = tensor.name || '';
            this._type = context.createTensorType(tensor.data_type, tensor.dims.map((dim) => dim), null);
            this._location = context.createLocation(tensor.data_location);
            this._values = data(tensor);
        }
    }

    get name() {
        return this._name;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
    }

    get value() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        const context = this._context();
        // console.log(context)
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        const value = this._decode(context, 0);
        // console.log(value)
        // console.log(onnx.Tensor._stringify(value, '', '    '))
        return onnx.Tensor._stringify(value, '', '    ');
    }

    _context() {
        const context = {};
        context.state = null;
        if (this._sparse) {
            context.state = 'Sparse data not implemented.';
            return context;
        }
        if (this._location !== 'default') {
            context.state = "Data '" + this._location + "' location not implemented.";
            return context;
        }
        const decode = (data) => {
            if (!data || Array.isArray(data) || ArrayBuffer.isView(data)) {
                return data;
            }
            const buffer = data.buffer;
            const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
            const type = data.type;
            data = undefined;
            switch (type) {
                case onnx.DataType.BOOL:
                    data = new Array(buffer.length);
                    for (let i = 0; i < buffer.length; i++) {
                        data[i] = view.getUint8(i) === 0 ? false : true;
                    }
                    break;
                case onnx.DataType.FLOAT16:
                    data = new Float32Array(buffer.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getFloat16(i << 1, true);
                    }
                    break;
                case onnx.DataType.FLOAT:
                    data = new Float32Array(buffer.length >> 2);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getFloat32(i << 2, true);
                    }
                    break;
                case onnx.DataType.DOUBLE:
                    data = new Float64Array(buffer.length >> 3);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getFloat64(i << 3, true);
                    }
                    break;
                case onnx.DataType.INT8:
                    data = new Int8Array(buffer.length);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt8(i, true);
                    }
                    break;
                case onnx.DataType.UINT8:
                    data = new Uint8Array(buffer.length);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint8(i, true);
                    }
                    break;
                case onnx.DataType.INT16:
                    data = new Int16Array(buffer.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt16(i << 1, true);
                    }
                    break;
                case onnx.DataType.UINT16:
                    data = new Uint16Array(buffer.length >> 1);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint16(i << 1, true);
                    }
                    break;
                case onnx.DataType.INT32:
                    data = new Int32Array(buffer.length >> 2);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt32(i << 2, true);
                    }
                    break;
                case onnx.DataType.UINT32:
                    data = new Uint32Array(buffer.length >> 2);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint32(i << 2, true);
                    }
                    break;
                case onnx.DataType.INT64:
                    data = new Array(buffer.length >> 3);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getInt64(i << 3, true);
                    }
                    break;
                case onnx.DataType.UINT64:
                    data = new Array(buffer.length >> 3);
                    for (let i = 0; i < data.length; i++) {
                        data[i] = view.getUint64(i << 3, true);
                    }
                    break;
            }
            return data;
        };
        this._values = decode(this._values);
        if (!this._values) {
            context.state = 'Tensor data is custom_add.';
            return context;
        }
        this._indices = decode(this._indices);
        context.values = this._values;
        context.indices = this._indices;
        context.index = 0;
        context.dataType = this.type.dataType;
        context.shape = this.type.shape.dimensions;
        context.data = function() {
            if (!this._data) {
                if (this.indices && this.values && this.indices.length === this.values.length) {
                    const size = context.shape.reduce((a, b) => a * b, 1);
                    const indices = this.indices;
                    const values = this.values;
                    const array = new values.constructor(size);
                    switch (this.dataType) {
                        case 'boolean':
                            array.fill(false);
                            break;
                        case 'int64':
                        case 'uint64':
                            break;
                    }
                    if (indices.length > 0) {
                        if (Object.prototype.hasOwnProperty.call(indices[0], 'low')) {
                            for (let i = 0; i < indices.length; i++) {
                                const index = indices[i];
                                array[index.high === 0 ? index.low : index.toNumber()] = values[i];
                            }
                        }
                        else {
                            for (let i = 0; i < indices.length; i++) {
                                array[indices[i]] = values[i];
                            }
                        }
                    }
                    this._data = array;
                }
                else {
                    this._data = this.values;
                }
            }
            return this._data;
        };
        return context;
    }

    _decode(context, dimension) {
        const shape = context.shape.length !== 0 ? context.shape : [ 1 ];
        const results = [];
        const size = shape[dimension];
        const data = context.data();
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.index > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(data[context.index++]);
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.index > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        // console.log(value, Array.isArray(value))
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => onnx.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (typeof value == 'string') {
            return indentation + value;
        }
        if (value == Infinity) {
            return indentation + 'Infinity';
        }
        if (value == -Infinity) {
            return indentation + '-Infinity';
        }
        if (isNaN(value)) {
            return indentation + 'NaN';
        }
        return indentation + value.toString();
    }
};

onnx.TensorType = class {

    constructor(dataType, shape, denotation) {
        this._dataType = dataType;
        this._shape = shape;
        this._denotation = denotation || null;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

onnx.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dim) => dim ? dim.toString() : '?').join(',') + ']';
    }
};

onnx.SequenceType = class {

    constructor(elementType, denotation) {
        this._elementType = elementType;
        this._denotation = denotation;
    }

    get elementType() {
        return this._elementType;
    }

    get dennotation() {
        return this._dennotation;
    }

    toString() {
        return 'sequence<' + this._elementType.toString() + '>';
    }
};

onnx.MapType = class {

    constructor(keyType, valueType, denotation) {
        this._keyType = keyType;
        this._valueType = valueType;
        this._denotation = denotation;
    }

    get keyType() {
        return this._keyType;
    }

    get valueType() {
        return this._valueType;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return 'map<' + this._keyType + ',' + this._valueType.toString() + '>';
    }
};

onnx.OpaqueType = class {

    constructor(domain, name) {
        this._domain = domain;
        this._name = name;
    }

    toString() {
        const name = (this._domain ? (this._domain + '.') : '') + this._name;
        return 'opaque<' + name + '>';
    }
};

onnx.Function = class {

    constructor(context, func) {
        this._name = func.name;
        this._domain = func.domain;
        this._description = func.doc_string;
        this._inputs = [];
        this._outputs = [];
        this._attributes = func.attribute.map((attribtue) => { return { name: attribtue }; });
        context = new onnx.GraphContext(context, func.node);
        func.input = func.input.map((input) => context.tensor(input));
        func.output = func.output.map((output) => context.tensor(output));
        context.push(func.node, func.input, func.output);
        this._nodes = context.pop();
        for (const input of func.input) {
            const argument = context.argument(input.name);
            if (!argument.initializer) {
                this._inputs.push(new onnx.Parameter(input.name, [ argument ]));
            }
        }
        for (const output of func.output) {
            const argument = context.argument(output.name);
            if (!argument.initializer) {
                this._outputs.push(new onnx.Parameter(output.name, [ argument ]));
            }
        }
    }

    get type() {
        return 'function';
    }

    get name() {
        return this._name;
    }

    get module() {
        return this._domain;
    }

    get description() {
        return this._description;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

    get nodes() {
        return this._nodes;
    }
};

onnx.GraphMetadata = class {

    constructor(metadata, imports) {
        this._metadata = metadata;
        this._imports = imports;
        this._cache = new Map();
        this._attributes = new Map();
        this._functions = new Map();
    }

    add(func) {
        if (!this._functions.has(func.module)) {
            this._functions.set(func.module, new Map());
        }
        const map = this._functions.get(func.module);
        if (map.has(func.name)) {
            throw new onnx.Error("Duplicate function identifier '" + func.module + '.' + func.name + "'.");
        }
        map.set(func.name, func);
    }

    type(name, domain) {
        domain = domain || 'ai.onnx';
        const key = domain + ':' + name;
        if (!this._cache.has(key)) {
            let value = this._metadata.type(name, domain, this._imports);
            if (!value) {
                if (this._functions.has(domain)) {
                    const map = this._functions.get(domain);
                    if (map.has(name)) {
                        value = map.get(name);
                    }
                }
            }
            this._cache.set(key, value);
        }
        return this._cache.get(key);
    }

    attribute(type, domain, name) {
        const key = domain + ':' + type + ':' + name;
        if (!this._attributes.has(key)) {
            const schema = this.type(type, domain);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    this._attributes.set(key, attribute);
                }
            }
            if (!this._attributes.has(key)) {
                this._attributes.set(key, null);
            }
        }
        return this._attributes.get(key);
    }
};

onnx.Metadata = class {

    static open(context) {
        if (onnx.Metadata._metadata) {
            return Promise.resolve(onnx.Metadata._metadata);
        }
        // return context.request('onnx-metadata.json', 'utf-8', null).then((data) => {
        return context.request('../static/onnx-metadata.json', 'utf-8', null).then((data) => {
            onnx.Metadata._metadata = new onnx.Metadata(data);
            return onnx.Metadata._metadata;
        }).catch(() => {
            onnx.Metadata._metadata = new onnx.Metadata(null);
            return onnx.Metadata._metadata;
        });
    }

    constructor(data) {
        this._map = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            for (const item of metadata) {
                if (!this._map.has(item.module)) {
                    this._map.set(item.module, new Map());
                }
                const map = this._map.get(item.module);
                if (!map.has(item.name)) {
                    map.set(item.name, []);
                }
                map.get(item.name).push(item);
            }
        }
    }

    type(name, domain, imports) {
        domain = domain || 'ai.onnx';
        let current = null;
        if (this._map.has(domain)) {
            const map = this._map.get(domain);
            if (map.has(name)) {
                for (const metadata of map.get(name)) {
                    const matchVersion = current ? current.version : -1;
                    const importVersion = imports.get(metadata.module) || 0;
                    // get the op with the version that are latest but doesn't exceed import version.
                    // TODO: take op domain into consideration to avoid op mismatch
                    // this can happen when different domains have the op with same name
                    // https://github.com/ZhangGe6/onnx-modifier/issues/136
                    if (metadata.version > matchVersion && metadata.version <= importVersion) {
                        current = metadata;
                    }
                }
            }
        }
        return current;
    }
};

onnx.Inference = class {

    constructor(nodes, outputs) {
        this._outputs = new Map();

        for (const node of nodes) {
            for (const output of node.output) {
                this._outputs.set(output.name, node);
            }
        }

        for (const output of outputs) {
            this._infer(output.name);
        }
    }

    _infer(output) {
        if (this._outputs.has(output)) {
            let hasInputShapes = true;
            const node = this._outputs.get(output);
            for (const input of node.input) {
                if (!input.type) {
                    this._infer(input);
                    if (!input.type) {
                        hasInputShapes = false;
                        break;
                    }
                }
            }
            if (hasInputShapes) {
                // continue
            }
        }
    }
};

onnx.DataLocation = {
    DEFAULT: 0,
    EXTERNAL: 1
};

onnx.DataType = {
    UNDEFINED: 0,
    FLOAT: 1,
    UINT8: 2,
    INT8: 3,
    UINT16: 4,
    INT16: 5,
    INT32: 6,
    INT64: 7,
    STRING: 8,
    BOOL: 9,
    FLOAT16: 10,
    DOUBLE: 11,
    UINT32: 12,
    UINT64: 13,
    COMPLEX64: 14,
    COMPLEX128: 15,
    BFLOAT16: 16
};

onnx.AttributeType = {
    UNDEFINED: 0,
    FLOAT: 1,
    INT: 2,
    STRING: 3,
    TENSOR: 4,
    GRAPH: 5,
    FLOATS: 6,
    INTS: 7,
    STRINGS: 8,
    TENSORS: 9,
    GRAPHS: 10,
    SPARSE_TENSOR: 11,
    SPARSE_TENSORS: 12,
    TYPE_PROTO: 13,
    TYPE_PROTOS: 14
};

onnx.AttributeTypeFromSchema = {

}

onnx.ModelContext = class {

    constructor(metadata, imageFormat) {
        this._metadata = metadata;
        this._imageFormat = imageFormat;
        this._graphs = new Map();
    }

    get metadata() {
        return this._metadata;
    }

    get imageFormat()  {
        return this._imageFormat;
    }

    graph(value) {
        if (!this._graphs.has(value)) {
            this._graphs.set(value, new onnx.Graph(this, value));
        }
        return this._graphs.get(value);
    }
};

onnx.GraphContext = class {
    // context here means ModelContext
    constructor(context, nodes) {
        this._context = context;
        this._decoder = new TextDecoder('utf-8');
        this._dataTypes = new Map(Object.entries(onnx.DataType).map((entry) => [ entry[1], entry[0].toLowerCase() ]));
        this._dataTypes.set(onnx.DataType.UNDEFINED, 'UNDEFINED');
        this._dataTypes.set(onnx.DataType.BOOL, 'boolean');
        this._dataTypes.set(onnx.DataType.FLOAT, 'float32');
        this._dataTypes.set(onnx.DataType.DOUBLE, 'float64');
        this._tensors = new Map();
        this._arguments = new Map();
        this._groups = new Map();
        this._nodes = [];
        for (const node of nodes) {
            node.input = node.input.map((name) => this.tensor(name));
            node.output = node.output.map((name) => this.tensor(name));
            node.param = {};
            for (const attribute of node.attribute) {
                if (attribute.type) {
                    continue;
                }
                if (attribute.ints && attribute.ints.length > 0) {
                    attribute.type = onnx.AttributeType.INTS;
                }
                else if (attribute.floats && attribute.floats.length > 0) {
                    attribute.type = onnx.AttributeType.FLOATS;
                }
                else if (attribute.strings && attribute.strings.length > 0) {
                    attribute.type = onnx.AttributeType.STRINGS;
                }
                else if (attribute.graphs && attribute.graphs.length > 0) {
                    attribute.type = onnx.AttributeType.GRAPHS;
                }
                else if (attribute.s && attribute.s.length > 0) {
                    attribute.type = onnx.AttributeType.STRING;
                }
                else if (Object.prototype.hasOwnProperty.call(attribute, 'f')) {
                    attribute.type = onnx.AttributeType.FLOAT;
                }
                else if (Object.prototype.hasOwnProperty.call(attribute, 'i')) {
                    attribute.type = onnx.AttributeType.INT;
                }
                else if (Object.prototype.hasOwnProperty.call(attribute, 't')) {
                    attribute.type = onnx.AttributeType.TENSOR;
                }
                else if (Object.prototype.hasOwnProperty.call(attribute, 'g')) {
                    attribute.type = onnx.AttributeType.GRAPH;
                }
                else if (Object.prototype.hasOwnProperty.call(attribute, 'sparse_tensor')) {
                    attribute.type =onnx.AttributeType.SPARSE_TENSOR;
                }
                else {
                    attribute.type = onnx.AttributeType.UNDEFINED;
                }
            }
        }
    }

    get metadata() {
        return this._context.metadata;
    }

    graph(name) {
        return this._context.graph(name);
    }

    tensor(name) {
        if (!this._tensors.has(name)) {
            this._tensors.set(name, { name: name });
        }
        return this._tensors.get(name);
    }

    group(name) {
        if (!this._groups.has(name)) {
            const path = name.split('/');
            if (path.length > 1) {
                path.pop();
                return this.group(path.join('/'));
            }
            this._groups.set(name, new Map([ [ '', [] ]]));
        }
        return this._groups.get(name);
    }

    argument(name, original_name) {
        const tensor = this.tensor(name);
        // console.log(tensor)
        const type = tensor.initializer ? tensor.initializer.type : tensor.type || null;
        this._arguments.set(original_name, new onnx.Argument(name, type, tensor.initializer, tensor.annotation, tensor.description, original_name));
        return this._arguments.get(original_name);
        // // if (!this._arguments.has(name)) {
        // if ((!this._arguments.has(name)) ||
        //     (this._arguments.has(name) && !this._arguments.get(name).original_name == original_name)
        // ) {
        //     const tensor = this.tensor(name);
        //     // console.log(name)
        //     // console.log(tensor)
        //     const type = tensor.initializer ? tensor.initializer.type : tensor.type || null;
        //     this._arguments.set(name, new onnx.Argument(name, type, tensor.initializer, tensor.annotation, tensor.description, original_name));
        // }
        // return this._arguments.get(name);
    }

    createType(type) {
        if (!type) {
            return null;
        }
        let denotation = '';
        switch (type.denotation) {
            case 'TENSOR':
                denotation = 'Tensor';
                break;
            case 'IMAGE':
                denotation = 'Image' + (this._context.imageFormat ? '(' + this._context.imageFormat.join(',') + ')' : '');
                break;
            case 'AUDIO':
                denotation = 'Audio';
                break;
            case 'TEXT':
                denotation = 'Text';
                break;
        }
        // switch (type.value) {
        //     case 'tensor_type': {
        //         const tensor_type = type.tensor_type;
        //         let shape = [];
        //         if (tensor_type.shape && tensor_type.shape.dim) {
        //             shape = tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value ? dim.dim_value : null);
        //         }
        //         return this.createTensorType(tensor_type.elem_type, shape, denotation);
        //     }
        //     case 'sparse_tensor_type': {
        //         const tensor_type = type.sparse_tensor_type;
        //         let shape = [];
        //         if (tensor_type.shape && tensor_type.shape.dim) {
        //             shape = tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value);
        //         }
        //         return this.createTensorType(tensor_type.elem_type, shape, denotation);
        //     }
        //     case 'map_type': {
        //         return this.createMapType(type.map_type.key_type, this.createType(type.map_type.value_type), denotation);
        //     }
        //     case 'sequence_type': {
        //         return new onnx.SequenceType(this.createType(type.sequence_type.elem_type), denotation);
        //     }
        //     case 'opaque_type': {
        //         return new onnx.OpaqueType(type.opaque_type.domain, type.opaque_type.name);
        //     }
        // }
        if (type.tensor_type) {
            const tensor_type = type.tensor_type;
            const shape = tensor_type.shape && tensor_type.shape.dim ? tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value ? dim.dim_value : null) : [];
            return this.createTensorType(tensor_type.elem_type, shape, null, denotation);
        } else if (type.sparse_tensor_type) {
            type = type.sparse_tensor_type;
            const shape = type.shape && type.shape.dim ? type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value ? dim.dim_value : null) : [];
            return this.createTensorType(type.elem_type, shape, 'sparse', denotation);
        } else if (type.map_type) {
            return this.createMapType(type.map_type.key_type, this.createType(type.map_type.value_type), denotation);
        } else if (type.sequence_type) {
            return new onnx.SequenceType(this.createType(type.sequence_type.elem_type), denotation);
        } else if (type.opaque_type) {
            return new onnx.OpaqueType(type.opaque_type.domain, type.opaque_type.name);
        } else if (type.optional_type) {
            return new onnx.OptionalType(this.createType(type.optional_type.elem_type), denotation);
        } else if (Object.keys(type).length == 0) {
            return null;
        }
        return null;
    }

    createTensorType(dataType, shape, denotation) {
        dataType = this.createDataType(dataType);
        return new onnx.TensorType(dataType, new onnx.TensorShape(shape), denotation);
    }

    createMapType(keyType, valueType, denotation) {
        keyType = this.createDataType(keyType);
        return new onnx.MapType(keyType, valueType, denotation);
    }

    createDataType(value) {
        return this._dataTypes.has(value) ? this._dataTypes.get(value) : this._dataTypes.get(onnx.DataType.UNDEFINED);
    }

    createLocation(value) {
        switch (value) {
            case onnx.DataLocation.DEFAULT: return 'default';
            case onnx.DataLocation.EXTERNAL: return 'external';
        }
        return 'UNDEFINED';
    }

    decodeText(value) {
        if (typeof value === 'string') {
            return value;
        }
        return this._decoder.decode(value);
    }

    push(nodes, inputs, outputs) {
        const inputMap = new Map();
        const outputMap = new Map();
        for (const node of nodes) {
            node.input.every((input) => inputMap.set(input.name, (inputMap.get(input) || 0) + 1));
            node.output.every((output) => outputMap.set(output.name, (outputMap.get(output) || 0) + 1));
        }
        inputs.every((input) => inputMap.delete(input.name));
        outputs.every((output) => outputMap.delete(output.name));
        nodes = nodes.filter((node) => {
            const constant = node &&
                node.op_type === 'Constant' &&
                node.attribute.length === 1 && node.attribute[0] &&
                node.input.length === 0 &&
                node.output.length === 1 && node.output[0] && inputMap.get(node.output[0].name) === 1 && outputMap.get(node.output[0].name) === 1;
            const attribute = constant ? node.attribute[0] : null;
            if (attribute && attribute.name === 'value' && attribute.type === onnx.AttributeType.TENSOR && attribute.t) {
                const tensor = this.tensor(node.output[0].name);
                tensor.initializer = new onnx.Tensor(this, attribute.t, 'Constant');
                return false;
            }
            else if (attribute && attribute.name === 'sparse_value' && attribute.type === onnx.AttributeType.SPARSE_TENSOR && attribute.sparse_tensor) {
                const tensor = this.tensor(node.output[0].name);
                tensor.initializer = new onnx.Tensor(this, attribute.sparse_tensor, 'Sparse Constant');
                return false;
            }
            return true;
        });
        for (let node of nodes) {
            const schema = this._context.metadata.type(node.op_type, node.domain);
            // console.log(node)     // NodeProto. It contains the uploaded model data
            // console.log(schema)   // get the corresponding schema of this node from Metadata
            const inputs = [];
            node.input = node.input || [];
            for (let i = 0; i < node.input.length; ) {
                const input = schema && schema.inputs && i < schema.inputs.length ? schema.inputs[i] : { name: i.toString() };
                const count = input.list ? node.input.length - i : 1;

                // slice the equal length of list from the upload model node
                // and convert them to Argument list
                // (instantiate a node here)
                const list = node.input.slice(i, i + count).map((input) => this.argument(input.name, input.name));
                inputs.push(new onnx.Parameter(input.name, list));
                i += count;
            }
            // console.log(inputs)
            const outputs = [];
            node.output = node.output || [];
            for (let i = 0; i < node.output.length; ) {
                const output = schema && schema.outputs && i < schema.outputs.length ? schema.outputs[i] : { name: i.toString() };
                const count = output.list ? node.output.length - i : 1;
                const list = node.output.slice(i, i + count).map((output) => this.argument(output.name, output.name));
                outputs.push(new onnx.Parameter(output.name, list));
                i += count;
            }
            // console.log(schema)
            // console.log(node)
            node = new onnx.Node(this, node.op_type, node.domain, node.name, node.doc_string, node.attribute, inputs, outputs);
            this._nodes.push(node);
            // console.log(node)

            // const path = (node.name || '').split('/');
            // path.pop();
            // this.group(path.join('/')).get('').push(node);
        }
    }

    pop() {
        /*
        const nodes = [];
        for (const entry of this._groups) {
            if (entry[0] === '') {
                for (const node of entry[1].get('')) {
                    nodes.push(node);
                }
                continue;
            }
            nodes.push(new onnx.Group(entry[0], entry[1]));
        }
        return nodes;
        */
        return this._nodes;
    }
};

onnx.Runtime = {};

onnx.Runtime.Reader = class {

    static open(stream, extension) {
        if (stream.length >= 8) {
            const buffer = stream.peek(Math.min(32, stream.length));
            const reader = flatbuffers.BinaryReader.open(buffer);
            const identifier = reader.identifier;
            if (identifier === 'ORTM') {
                return new onnx.Runtime.Reader(stream);
            }
            if (extension === 'ort') {
                const signature = [ 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ];
                if (signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
                    return new onnx.Runtime.Reader(stream);
                }
            }
        }
        return null;
    }

    constructor(stream) {
        this._stream = stream;
    }

    read() {
        this._graphs = new Set();
        const reader = flatbuffers.BinaryReader.open(this._stream);
        const session = onnx.schema.InferenceSession.create(reader);
        const model = session.model;
        const graph = model.graph;
        graph.doc_string = model.graph_doc_string;
        delete model.graph_doc_string;
        this._graph(graph);
        return model;
    }

    _graph(graph) {
        if (this._graphs.has(graph)) {
            return;
        }
        this._graphs.add(graph);
        graph.name = this._graphs.size.toString();
        graph.node = graph.nodes.map((node) => {
            this._node(node);
            return node;
        });
        delete graph.nodes;
        graph.input = graph.inputs.map((input) => {
            return { name: input };
        });
        delete graph.inputs;
        graph.output = graph.outputs.map((output) => {
            return { name: output };
        });
        delete graph.outputs;
        graph.value_info = graph.node_args;
        delete graph.node_args;
        graph.initializer = graph.initializers.map((tensor) => {
            tensor.data_location = onnx.DataLocation.DEFAULT;
            return tensor;
        });
        delete graph.initializers;
        graph.sparse_initializer = graph.sparse_initializers.map((tensor) => {
            tensor.values.data_location = onnx.DataLocation.DEFAULT;
            tensor.indices.data_location = onnx.DataLocation.DEFAULT;
            return tensor;
        });
        delete graph.sparse_initializers;
    }

    _node(node) {
        node.input = node.inputs;
        node.output = node.outputs;
        node.attribute = node.attributes.map((attribute) => {
            switch (attribute.type) {
                case onnx.AttributeType.GRAPH:
                    this._graph(attribute.g);
                    break;
                case onnx.AttributeType.GRAPHS:
                    for (const graph of attribute.graphs) {
                        this._graph(graph);
                    }
                    break;
            }
            return attribute;
        });
        delete node.inputs;
        delete node.outputs;
        delete node.attributes;
    }
};

onnx.Text = {};

onnx.Text.Reader = class {

    static open(stream) {
        try {
            if (stream.length > 0 && stream.peek(1)[0] < 0x80 || stream.peek(1)[0] >= 0xFE) {
                const reader = text.Reader.open(stream);
                const lines = [];
                for (let i = 0; i < 32; i++) {
                    const line = reader.read();
                    if (line === undefined) {
                        break;
                    }
                    lines.push(line);
                }
                const content = lines.join('\n');
                if (/^\s*<\s*ir_version\s*:/m.exec(content) ||
                    /^\s*[a-zA-Z][a-zA-Z0-9]*\s*\(.*\)\s=>\s\(/m.exec(content)) {
                    return new onnx.Text.Reader(stream);
                }
            }
        }
        catch (err) {
            // continue regardless of error
        }
        return null;
    }

    constructor(stream) {
        this._stream = stream;
        this._dataTypes = new Map([
            [ 'float', 1 ], [ 'uint8', 2 ], [ 'int8', 3 ], [ 'uint16', 4 ],
            [ 'int16', 5 ], [ 'int32', 6 ], [ 'int64', 7 ], [ 'string', 8 ],
            [ 'bool', 9 ], [ 'float16', 10 ], [ 'double', 11 ], [ 'uint32', 12 ],
            [ 'uint64', 13 ], [ 'complex64', 14 ], [ 'complex128', 15 ], [ 'bfloat16', 16 ]
        ]);
        this._attributeTypes = new Map([
            [ 'float', 1 ], [ 'int', 2 ], [ 'string', 3 ],
            [ 'tensor', 4 ], [ 'graph', 5 ], [ 'sparse_tensor', 11 ], [ 'type_proto', 13 ],
            [ 'floats', 6 ], [ 'ints', 7 ], [ 'strings', 8 ],
            [ 'tensors', 9 ], [ 'graphs', 10 ], [ 'sparse_tensors', 12 ], [ 'type_protos', 14 ]
        ]);
    }

    read() {
        const decoder = text.Decoder.open(this._stream);
        this._decoder = decoder;
        this._position = 0;
        this._char = decoder.decode();
        return this._model();
    }

    _seek(position) {
        this._decoder.position = position;
        this._char = '';
        this._next();
    }

    _model() {
        this._whitespace();
        const model = new onnx.proto.ModelProto();
        if (this._match('<')) {
            do {
                const keyword = this._identifier();
                this._expect(':');
                switch (keyword) {
                    case 'ir_version':
                    case 'model_version':
                        model[keyword] = this._integer();
                        break;
                    case 'opset_import':
                        model[keyword] = this._operatorSetId();
                        break;
                    case 'producer_name':
                    case 'producer_version':
                    case 'domain':
                    case 'doc_string':
                        model[keyword] = this._string();
                        break;
                    case 'metadata_props':
                        this._expect('[');
                        if (!this._match(']')) {
                            do {
                                const entry = new onnx.proto.StringStringEntryProto();
                                entry.key = this._string();
                                this._expect(':');
                                entry.value = this._string();
                                model.metadata_props.push(entry);
                            } while (this._match(','));
                            this._expect(']');
                        }
                        break;
                    default:
                        this._throw("Unknown keyword '" + keyword + "'.");
                        break;
                }
            } while (this._match(','));
            this._expect('>');
        }
        model.graph = this._graph();
        this._whitespace();
        while (this._char !== undefined) {
            const func = this._function();
            if (func) {
                model.functions.push(func);
            }
            this._whitespace();
        }
        return model;
    }

    _graph() {
        const graph = new onnx.proto.GraphProto();
        graph.name = this._identifier();
        if (this._match('(')) {
            if (!this._match(')')) {
                do {
                    const valueInfo = this._valueInfo();
                    if (this._match('=')) {
                        const tensor = this._tensor(valueInfo.type);
                        tensor.name = valueInfo.name;
                        graph.initializer.push(tensor);
                    }
                    graph.input.push(valueInfo);
                }
                while (this._match(','));
                this._expect(')');
            }
        }
        this._expect('=>');
        graph.output = this._valueInfoList();
        if (this._match('<')) {
            if (!this._match('>')) {
                do {
                    const valueInfo = this._valueInfo();
                    if (this._match('=')) {
                        const tensor = this._tensor(valueInfo.type);
                        tensor.name = valueInfo.name;
                        graph.initializer.push(tensor);
                    }
                    else {
                        graph.value_info.push(valueInfo);
                    }
                }
                while (this._match(','));
                this._expect('>');
            }
        }
        graph.node = this._nodeList();
        return graph;
    }

    _nodeList() {
        const list = [];
        this._expect('{');
        while (!this._match('}')) {
            list.push(this._node());
        }
        return list;
    }

    _node() {
        const node = new onnx.proto.NodeProto();
        node.output = this._identifierList();
        this._expect('=');
        let identifier = this._identifier();
        let domain = '';
        while (this._match('.')) {
            if (domain) {
                domain += '.';
            }
            domain += identifier;
            identifier = this._identifier();
        }
        node.domain = domain;
        node.op_type = identifier;
        node.attribute = this._attributeList();
        this._expect('(');
        node.input = this._identifierList();
        this._expect(')');
        if (!node.attribute || node.attribute.length === 0) {
            node.attribute = this._attributeList();
        }
        return node;
    }

    _attributeList() {
        const list = [];
        if (this._match('<')) {
            do {
                list.push(this._attribute());
            }
            while (this._match(','));
            this._expect('>');
        }
        return list;
    }

    _attribute() {
        const attribute = new onnx.proto.AttributeProto();
        attribute.name = this._identifier();
        if (this._match(':')) {
            const type = this._identifier();
            if (!this._attributeTypes.has(type)) {
                this._throw("Unexpected attribute type '" + type + "'.");
            }
            attribute.type = this._attributeTypes.get(type);
        }
        this._expect('=');
        if (this._match('[')) {
            const list = [];
            do {
                list.push(this._literal());
            }
            while (this._match(','));
            this._expect(']');
            if (list.every((value) => typeof value === 'string')) {
                attribute.type = onnx.AttributeType.STRINGS;
                attribute.strings = list;
            }
            else if (list.every((value) => typeof value === 'number' && Number.isInteger(value))) {
                attribute.type = onnx.AttributeType.INTS;
                attribute.ints = list;
            }
            else if (list.every((value) => typeof value === 'number')) {
                attribute.type = onnx.AttributeType.FLOATS;
                attribute.floats = list;
            }
            else {
                this._throw("Unexpected value '" + JSON.stringify(list) + "'.");
            }
        }
        else {
            if ((this._char >= 'a' && this._char <= 'z') || (this._char >= 'A' && this._char <= 'Z') || this._char === '_') {
                const identifier = this._identifier();
                if (this._dataTypes.has(identifier)) {
                    attribute.type = onnx.AttributeType.TENSOR;
                    if (!this._dataTypes.has(identifier)) {
                        this._throw("Unexpected type '" + identifier + "'.");
                    }
                    const type = this._type(this._dataTypes.get(identifier));
                    if (!type.tensor_type.elem_type) {
                        this._throw('Expected tensor data type.');
                    }
                    if (!type.tensor_type.shape || !type.tensor_type.shape.dim) {
                        this._throw('Expected tensor shape.');
                    }
                    attribute.t = this._tensor(type);
                }
                else {
                    attribute.type = onnx.AttributeType.GRAPH;
                    attribute.g = this._graph();
                }
            }
            else if (this._match('@')) {
                attribute.ref_attr_name = this._identifier();
            }
            else {
                const value = this._literal();
                switch (typeof value) {
                    case 'number':
                        if (Number.isInteger(value)) {
                            attribute.type = onnx.AttributeType.INT;
                            attribute.i = value;
                        }
                        else {
                            attribute.type = onnx.AttributeType.FLOAT;
                            attribute.f = value;
                        }
                        break;
                    case 'string':
                        attribute.type = onnx.AttributeType.STRING;
                        attribute.s = value;
                        break;
                    default: {
                        this._throw("Unexpected value '" + JSON.stringify(value) + "'.");
                    }
                }
            }
        }
        return attribute;
    }

    _valueInfoList() {
        const list = [];
        this._expect('(');
        if (!this._match(')')) {
            do {
                list.push(this._valueInfo());
            } while (this._match(','));
            this._expect(')');
        }
        return list;
    }

    _valueInfo() {
        const valueInfo = new onnx.proto.ValueInfoProto();
        let identifier = this._identifier();
        if (this._dataTypes.has(identifier)) {
            valueInfo.type = this._type(this._dataTypes.get(identifier));
            identifier = this._identifier();
        }
        valueInfo.name = identifier;
        return valueInfo;
    }

    _type(elem_type) {
        const type = new onnx.proto.TypeProto();
        type.tensor_type = new onnx.proto.TypeProto.Tensor();
        type.tensor_type.elem_type = elem_type;
        if (this._match('[')) {
            if (!this._match(']')) {
                type.tensor_type.shape = this._shape();
                this._expect(']');
            }
        }
        else {
            type.tensor_type.shape = new onnx.proto.TensorShapeProto();
        }
        return type;
    }

    _shape() {
        const shape = new onnx.proto.TensorShapeProto();
        do {
            const dimension = new onnx.proto.TensorShapeProto.Dimension();
            if (!this._match('?')) {
                const identifier = this._identifier(true);
                if (identifier) {
                    dimension.dim_param = identifier;
                }
                else {
                    dimension.dim_value = this._integer();
                }
            }
            shape.dim.push(dimension);
        }
        while (this._match(','));
        return shape;
    }

    _tensor(type) {
        const tensor = new onnx.proto.TensorProto();
        if (!type.tensor_type || !type.tensor_type.elem_type) {
            this._throw('Expected tensor type.');
        }
        if (!type.tensor_type.shape || !type.tensor_type.shape.dim || !type.tensor_type.shape.dim.every((dim) => dim.dim_value)) {
            this._throw('Expected numeric tensor shape.');
        }
        const elem_type = type.tensor_type.elem_type;
        tensor.data_type = elem_type;
        tensor.dims = type.tensor_type.shape.dim.map((dim) => dim.dim_value);
        this._match('=');
        this._expect('{');
        if (!this._match('}')) {
            do {
                switch (elem_type) {
                    case onnx.DataType.INT8:
                    case onnx.DataType.INT16:
                    case onnx.DataType.INT32:
                    case onnx.DataType.UINT8:
                    case onnx.DataType.UINT16:
                    case onnx.DataType.BOOL:
                        tensor.int32_data.push(this._integer());
                        break;
                    case onnx.DataType.INT64:
                        tensor.int64_data.push(this._integer());
                        break;
                    case onnx.DataType.UINT32:
                    case onnx.DataType.UINT64:
                        tensor.uint64_data.push(this._integer());
                        break;
                    case onnx.DataType.FLOAT:
                        tensor.float_data.push(this._float());
                        break;
                    case onnx.DataType.DOUBLE:
                        tensor.double_data.push(this._float());
                        break;
                    case onnx.DataType.STRING:
                        tensor.string_data.push(this.string());
                        break;
                    default:
                        return this._throw("Unsupported tensor element type '" + elem_type.toString() + "'.");
                }
            } while (this._match(','));
            this._expect('}');
        }
        return tensor;
    }

    _function() {
        const func = new onnx.proto.FunctionProto();
        if (this._match('<')) {
            do {
                const keyword = this._identifier();
                this._expect(':');
                switch (keyword) {
                    case 'opset_import':
                        func[keyword] = this._operatorSetId();
                        break;
                    case 'domain':
                    case 'doc_string':
                        func[keyword] = this._string();
                        break;
                    default:
                        this._throw("Unknown keyword '" + keyword + "'.");
                        break;
                }
            }
            while (this._match(','));
            this._expect('>');
        }
        func.name = this._identifier();
        if (this._match('<')) {
            func.attribute = this._identifierList();
            this._expect('>');
        }
        if (this._match('(')) {
            func.input = this._identifierList();
            this._expect(')');
        }
        this._expect('=>');
        if (this._match('(')) {
            func.output = this._identifierList();
            this._expect(')');
        }
        func.node = this._nodeList();
        return func;
    }

    _identifierList() {
        const list = [];
        const identifier = this._identifier(true);
        if (identifier) {
            list.push(identifier);
            while (this._match(',')) {
                list.push(this._identifier());
            }
        }
        return list;
    }

    _identifier(optional) {
        this._whitespace();
        const value = [];
        if ((this._char >= 'a' && this._char <= 'z') || (this._char >= 'A' && this._char <= 'Z')) {
            value.push(this._char);
            this._next();
            while ((this._char >= 'a' && this._char <= 'z') || (this._char >= 'A' && this._char <= 'Z') || (this._char >= '0' && this._char <= '9') || this._char === '_') {
                value.push(this._char);
                this._next();
            }
        }
        if (optional !== true && value.length == 0) {
            this._throw('Identifier expected.');
        }
        return value.join('');
    }

    _literal() {
        this._whitespace();
        let decimal_point = false;
        if (this._char === '"') {
            const value = [];
            this._next();
            while (this._char !== undefined && this._char !== '"') {
                value.push(this._char);
                this._next();
            }
            if (this._char !== undefined) {
                this._next();
            }
            return value.join('');
        }
        else if ((this._char >= '0' && this._char <= '9') || this._char === '-') {
            const value = [ this._char ];
            this._next();
            while ((this._char >= '0' && this._char <= '9') || this._char === '.') {
                if (this._char === '.') {
                    if (decimal_point) {
                        this._throw();
                    }
                    decimal_point = true;
                }
                value.push(this._char);
                this._next();
            }
            if (value.length === 0) {
                this._throw('Value expected.');
            }
            if (this._char === 'e' || this._char === 'E') {
                decimal_point = true;
                value.push(this._char);
                this._next();
                if (this._char === '+' || this._char === '-') {
                    value.push(this._char);
                    this._next();
                }
                while ((this._char >= '0' && this._char <= '9')) {
                    value.push(this._char);
                    this._next();
                }
            }
            return decimal_point ? Number.parseFloat(value.join('')) : Number.parseInt(value.join(''), 10);
        }
        return undefined;
    }

    _integer() {
        const value = this._literal();
        if (!Number.isInteger(value)) {
            this._throw('Integer value expected.');
        }
        return value;
    }

    _float() {
        const value = this._literal();
        if (typeof value !== 'number') {
            this._throw('Float value expected.');
        }
        return value;
    }

    _string() {
        const value = this._literal();
        if (typeof value !== 'string') {
            this._throw('String value expected.');
        }
        return value;
    }

    _operatorSetId() {
        const list = [];
        this._expect('[');
        if (!this._match(']')) {
            do {
                const value = new onnx.proto.OperatorSetIdProto();
                value.domain = this._string();
                this._expect(':');
                value.version = this._integer();
                list.push(value);
            }
            while (this._match(','));
            this._expect(']');
        }
        return list;
    }

    _match(value) {
        this._whitespace();
        if (this._char !== value[0]) {
            return false;
        }
        if (value.length === 1) {
            this._next();
            return true;
        }
        const position = this._position;
        for (let i = 0; i < value.length; i++) {
            if (this._char !== value[i]) {
                this._seek(position);
                return false;
            }
            this._next();
        }
        return true;
    }

    _expect(value) {
        if (!this._match(value)) {
            this._unexpected();
        }
        return true;
    }

    _whitespace() {
        for (;;) {
            while (this._char === ' ' || this._char === '\n' || this._char === '\r' || this._char === '\t') {
                this._next();
            }
            if (this._char === undefined || this._char !== '#') {
                break;
            }
            while (this._char !== undefined && this._char !== '\n') {
                this._next();
            }
        }
    }

    _next() {
        if (this._char === undefined) {
            this._unexpected();
        }
        this._position = this._decoder.position;
        this._char = this._decoder.decode();
    }

    _unexpected() {
        let c = this._char;
        if (c === undefined) {
            throw new onnx.Error('Unexpected end of input.');
        }
        else if (c === '"') {
            c = 'string';
        }
        else if ((c >= '0' && c <= '9') || c === '-') {
            c = 'number';
        }
        else {
            if (c < ' ' || c > '\x7F') {
                const name = Object.keys(this._escape).filter((key) => this._escape[key] === c);
                c = (name.length === 1) ? '\\' + name : '\\u' + ('000' + c.charCodeAt(0).toString(16)).slice(-4);
            }
            c = "token '" + c + "'";
        }
        this._throw('Unexpected ' + c);
    }

    _throw(message) {
        throw new onnx.Error(message.replace(/\.$/, '') + this._location());
    }

    _location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c;
        do {
            if (this._decoder.position === this._position) {
                return ' at ' + line.toString() + ':' + column.toString() + '.';
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            }
            else {
                column++;
            }
        }
        while (c !== undefined);
        return ' at ' + line.toString() + ':' + column.toString() + '.';
    }
};

onnx.JsonReader = class {

    static open(context) {
        const obj = context.open('json');
        if (obj && (obj.irVersion !== undefined || (obj.graph && Array.isArray(obj.graph.node)))) {
            return new onnx.JsonReader(obj);
        }
        return null;
    }

    constructor(obj) {
        this.model = obj;
        this._attributeTypes = new Map(Object.entries(onnx.AttributeType));
    }

    async read() {
        const tensor_shape = (value) => {
            if (Array.isArray(value.dim)) {
                for (const dimension of value.dim) {
                    if (dimension.dimValue !== undefined) {
                        dimension.dim_value = parseInt(dimension.dimValue, 10);
                        delete dimension.dimValue;
                    } else if (dimension.dimParam !== undefined) {
                        dimension.dim_param = dimension.dimParam;
                        delete dimension.dimParam;
                    }
                }
            }
            return value;
        };
        const tensor_type = (value) => {
            value.elem_type = value.elemType;
            delete value.elemType;
            if (value.shape) {
                value.shape = tensor_shape(value.shape);
            }
            return value;
        };
        /* eslint-disable no-use-before-define */
        const optional_type = (value) => {
            value.elem_type = type(value.elemType);
            delete value.elemType;
            return value;
        };
        const sequence_type = (value) => {
            value.elem_type = type(value.elemType);
            delete value.elemType;
            return value;
        };
        const map_type = (value) => {
            value.key_type = value.keyType;
            delete value.keyType;
            value.value_type = type(value.valueType);
            delete value.valueType;
            return value;
        };
        const sparse_tensor_type = (value) => {
            value.elem_type = value.elemType;
            delete value.elemType;
            if (value.shape) {
                value.shape = tensor_shape(value.shape);
            }
            return value;
        };
        const type = (value) => {
            if (value.tensorType) {
                value.tensor_type = tensor_type(value.tensorType);
                delete value.tensorType;
            } else if (value.sequenceType) {
                value.sequence_type = sequence_type(value.sequenceType);
                delete value.sequenceType;
            } else if (value.optionalType) {
                value.optional_type = optional_type(value.optionalType);
                delete value.optionalType;
            } else if (value.mapType) {
                value.map_type = map_type(value.mapType);
                delete value.mapType;
            } else if (value.sparseTensorType) {
                value.sparse_tensor_type = sparse_tensor_type(value.sparseTensorType);
                delete value.sparseTensorType;
            } else {
                throw new onnx.Error("Unsupported ONNX JSON type '" + JSON.stringify(Object.keys(value)) + "'.");
            }
            return value;
        };
        const tensor = (value) => {
            value.data_type = value.dataType;
            value.dims = Array.isArray(value.dims) ? value.dims.map((dim) => parseInt(dim, 10)) : [];
            delete value.dataType;
            if (value.rawData !== undefined) {
                value.data_location = onnx.DataLocation.DEFAULT;
                const data = atob(value.rawData);
                const length = data.length;
                const array = new Uint8Array(length);
                for (let i = 0; i < length; i++) {
                    array[i] = data[i].charCodeAt(0);
                }
                value.raw_data = array;
                delete value.rawData;
            } else if (Array.isArray(value.floatData)) {
                value.data_location = onnx.DataLocation.DEFAULT;
                value.float_data = value.floatData;
                delete value.floatData;
            } else if (Array.isArray(value.int32Data)) {
                value.data_location = onnx.DataLocation.DEFAULT;
                value.int32_data = value.int32Data;
                delete value.int32Data;
            } else if (Array.isArray(value.int64Data)) {
                value.data_location = onnx.DataLocation.DEFAULT;
                value.int64_data = value.int64Data.map((value) => parseInt(value, 10));
                delete value.int64Data;
            } else {
                throw new onnx.Error("Unsupported ONNX JSON tensor data '" + JSON.stringify(value.data_type) + ".");
            }
            return value;
        };
        const sparse_tensor = (value) => {
            value.indices = tensor(value.indices);
            value.values = tensor(value.values);
            return value;
        };
        const attribute = (value) => {
            if (value.type && this._attributeTypes.has(value.type)) {
                value.type = this._attributeTypes.get(value.type);
            }
            if (value.refAttrName) {
                value.ref_attr_name = value.refAttrName;
                delete value.refAttrName;
            } else if (value.type === onnx.AttributeType.FLOATS || Array.isArray(value.floats)) {
                value.floats = value.floats.map((value) => parseFloat(value));
            } else if (value.type === onnx.AttributeType.INTS || Array.isArray(value.ints)) {
                value.ints = value.ints.map((value) => parseInt(value, 10));
            } else if (value.type === onnx.AttributeType.STRINGS || Array.isArray(value.strings)) {
                value.strings = value.strings.map((value) => atob(value));
            } else if (value.type === onnx.AttributeType.TENSORS || Array.isArray(value.tensors)) {
                value.tensors = value.tensors.map((value) => tensor(value));
            } else if (value.type === onnx.AttributeType.GRAPHS || Array.isArray(value.graphs)) {
                value.graphs = value.graphs.map((value) => graph(value));
            } else if (value.type === onnx.AttributeType.SPARSE_TENSORS || Array.isArray(value.sparseTensors)) {
                value.sparse_tensors = value.sparseTensors.map((value) => sparse_tensor(value));
                delete value.sparseTensors;
            } else if (value.type === onnx.AttributeType.FLOAT || value.f !== undefined) {
                value.f = parseFloat(value.f);
            } else if (value.type === onnx.AttributeType.INT || value.i !== undefined) {
                value.i = parseInt(value.i, 10);
            } else if (value.type === onnx.AttributeType.STRING || value.s !== undefined) {
                value.s = atob(value.s);
            } else if (value.type === onnx.AttributeType.TENSOR || value.t !== undefined) {
                value.t = tensor(value.t);
            } else if (value.type === onnx.AttributeType.GRAPH || value.g !== undefined) {
                value.g = graph(value.g);
            } else if (value.type === onnx.AttributeType.SPARSE_TENSOR || value.sparseTensor !== undefined) {
                value.sparse_tensor = sparse_tensor(value.sparseTensor);
                delete value.sparseTensor;
            } else {
                throw new onnx.Error("Unsupported ONNX JSON attribute type '" + JSON.stringify(value.type) + "'.");
            }
            return value;
        };
        const node = (value) => {
            value.op_type = value.opType;
            delete value.opType;
            value.input = Array.isArray(value.input) ? value.input : [];
            value.output = Array.isArray(value.output) ? value.output : [];
            value.attribute = Array.isArray(value.attribute) ? value.attribute.map((value) => attribute(value)) : [];
            return value;
        };
        const value_info = (value) => {
            value.type = type(value.type);
            return value;
        };
        const operator_set = (value) => {
            value.version = parseInt(value.version, 10);
            return value;
        };
        const graph = (value) => {
            value.node = value.node.map((value) => node(value));
            value.initializer = Array.isArray(value.initializer) ? value.initializer.map((value) => tensor(value)) : [];
            value.sparse_initializer = Array.isArray(value.sparseInitializer) ? value.sparseInitializer.map((value) => sparse_tensor(value)) : [];
            value.value_info = Array.isArray(value.valueInfo) ? value.valueInfo.map((value) => value_info(value)) : [];
            value.input = Array.isArray(value.input) ? value.input.map((value) => value_info(value)) : [];
            value.output = Array.isArray(value.output) ? value.output.map((value) => value_info(value)) : [];
            return value;
        };
        const func = (value) => {
            value.node = value.node.map((value) => node(value));
            value.input = Array.isArray(value.input) ? value.input : [];
            value.output = Array.isArray(value.output) ? value.output : [];
            value.attribute = Array.isArray(value.attribute) ? value.attribute : [];
            value.attribute_proto = Array.isArray(value.attributeProto) ? value.attributeProto.map((value) => attribute(value)) : [];
            delete value.attributeProto;
            if (value.docString) {
                value.doc_string = value.docString;
                delete value.docString;
            }
            return value;
        };
        /* eslint-enable no-use-before-define */
        this.model.ir_version = parseInt(this.model.irVersion, 10);
        delete this.model.irVersion;
        if (this.model.version !== undefined) {
            this.model.version = parseInt(this.model.version, 10);
        }
        if (this.model.producerName) {
            this.model.producer_name = this.model.producerName;
            delete this.model.producerName;
        }
        if (this.model.producerVersion) {
            this.model.producer_version = this.model.producerVersion;
            delete this.model.producerVersion;
        }
        if (this.model.modelVersion) {
            this.model.model_version = parseInt(this.model.modelVersion, 10);
            delete this.model.modelVersion;
        }
        if (this.model.docString) {
            this.model.doc_string = this.model.docString;
            delete this.model.docString;
        }
        this.model.graph = graph(this.model.graph);
        if (Array.isArray(this.model.opsetImport)) {
            this.model.opset_import = this.model.opsetImport.map((value) => operator_set(value));
            delete this.model.opsetImport;
        }
        if (Array.isArray(this.model.metadataProps)) {
            this.model.metadata_props = this.model.metadataProps;
            delete this.model.metadataProps;
        }
        if (Array.isArray(this.model.functions)) {
            this.model.functions = this.model.functions.map((value) => func(value));
        }
        this.format = 'ONNX JSON' + (this.model.ir_version ? ' v' + this.model.ir_version.toString() : '');
    }
};

onnx.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading ONNX model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = onnx.ModelFactory;
}
