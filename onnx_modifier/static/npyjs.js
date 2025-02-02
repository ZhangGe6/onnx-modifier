// https://github.com/aplbrain/npyjs

// import fetch from 'cross-fetch';

var npyjs = npyjs || {};

npyjs.Npyjs = class {

    constructor(opts) {
        if (opts) {
            console.error([
                "No arguments accepted to npyjs constructor.",
                "For usage, go to https://github.com/jhuapl-boss/npyjs."
            ].join(" "));
        }

        this.dtypes = {
            "<u1": {
                name: "uint8",
                size: 8,
                arrayConstructor: Uint8Array,
            },
            "|u1": {
                name: "uint8",
                size: 8,
                arrayConstructor: Uint8Array,
            },
            "<u2": {
                name: "uint16",
                size: 16,
                arrayConstructor: Uint16Array,
            },
            "|i1": {
                name: "int8",
                size: 8,
                arrayConstructor: Int8Array,
            },
            "<i2": {
                name: "int16",
                size: 16,
                arrayConstructor: Int16Array,
            },
            "<u4": {
                name: "uint32",
                size: 32,
                arrayConstructor: Int32Array,
            },
            "<i4": {
                name: "int32",
                size: 32,
                arrayConstructor: Int32Array,
            },
            "<u8": {
                name: "uint64",
                size: 64,
                arrayConstructor: BigUint64Array,
            },
            "<i8": {
                name: "int64",
                size: 64,
                arrayConstructor: BigInt64Array,
            },
            "<f4": {
                name: "float32",
                size: 32,
                arrayConstructor: Float32Array
            },
            "<f8": {
                name: "float64",
                size: 64,
                arrayConstructor: Float64Array
            },
        };
    }

    parse(arrayBufferContents) {
        // const version = arrayBufferContents.slice(6, 8); // Uint8-encoded
        const headerLength = new DataView(arrayBufferContents.slice(8, 10)).getUint8(0);
        const offsetBytes = 10 + headerLength;

        const hcontents = new TextDecoder("utf-8").decode(
            new Uint8Array(arrayBufferContents.slice(10, 10 + headerLength))
        );
        const header = JSON.parse(
            hcontents
                .toLowerCase() // True -> true
                .replace(/'/g, '"')
                .replace("(", "[")
                .replace(/,*\),*/g, "]")
        );
        const shape = header.shape;
        const dtype = this.dtypes[header.descr];
        const nums = new dtype["arrayConstructor"](
            arrayBufferContents,
            offsetBytes
        );
        return {
            dtype: dtype.name,
            data: nums,
            shape,
            fortranOrder: header.fortran_order
        };
    }

    is_numerix(x){
        return /[0-9]/.test(x) || ["+", "-", "e", "."].includes(x);
    }

    format_np(nums, shape) {
        // convert numeric into string format
        for (var i = shape.length - 1; i > -1; i -= 1) {
            var segs = [];
            var seg_len = shape[i];
            var seg_str = "["
            for (var j = 0; j < nums.length; j += seg_len) {
                seg_str = "[" + nums.slice(j, j + seg_len).map((x) => x.toString()) + "]" ;
                segs.push(seg_str);
            }
            nums = segs;
        }
        var nums_str = nums[0];

        // now nums is in string format
        // get indent for each character
        var indent_num = new Array();
        var cur_indent = 0;
        for (var i = 0; i < nums_str.length; i += 1) {
            if (nums_str[i] == "[") {
                indent_num.push(cur_indent);
                cur_indent += 1;
            } else if (nums_str[i] == "]") {
                cur_indent -= 1;
                indent_num.push(cur_indent);
            } else{
                indent_num.push(cur_indent);
            }
        }

        // console.log(nums_str);
        // console.log(indent_num);
        // get the formated string
        var res_str = "";
        for (var i = 0; i < nums_str.length; i += 1) {
            var value = nums_str[i];
            var indent = indent_num[i];
            if (value == "[") {
                res_str += "    ".repeat(indent) + value + "\n";
            } else if (value == ",") {
                res_str += value + "\n";
            } else if (value == "]") {
                res_str += "    ".repeat(indent) + value;
                if (!(i + 1 < nums_str.length && nums_str[i + 1] == ",")) {
                    res_str += "\n";
                }
            } else { // is numeric
                if (i - 1 > 0 && !this.is_numerix(nums_str[i - 1])) {
                    res_str += "    ".repeat(indent);
                }
                res_str += value;
                if (i + 1 < nums_str.length &&
                    !(this.is_numerix(nums_str[i + 1]) || nums_str[i + 1] == ",")) {
                    res_str += "\n";
                }
            }
        }
        return res_str;
    }

    async load(filename, callback, fetchArgs) {
        /*
        Loads an array from a stream of bytes.
        */
        fetchArgs = fetchArgs || {};
        let arrayBuf;
        // If filename is ArrayBuffer
        if (filename instanceof ArrayBuffer) {
            arrayBuf = filename;
        }
        // If filename is a file path
        else {
            const resp = await fetch(filename, { ...fetchArgs });
            arrayBuf = await resp.arrayBuffer();
        }
        const result = this.parse(arrayBuf);
        if (callback) {
            return callback(result);
        }
        return result;
    }
}

// export default npyjs;
if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Npyjs = npyjs.Npyjs;
}

