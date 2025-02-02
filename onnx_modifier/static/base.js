
var base = base || {};

base.Int64 = class Int64 {

    constructor(low, high) {
        this.low = low | 0;
        this.high = high | 0;
    }

    static create(value) {
        if (isNaN(value)) {
            return base.Int64.zero;
        }
        if (value <= -9223372036854776000) {
            return base.Int64.min;
        }
        if (value + 1 >= 9223372036854776000) {
            return base.Int64.max;
        }
        if (value < 0) {
            return base.Int64.create(-value).negate();
        }
        return new base.Int64((value % 4294967296) | 0, (value / 4294967296));
    }

    get isZero() {
        return this.low === 0 && this.high === 0;
    }

    get isNegative() {
        return this.high < 0;
    }

    negate() {
        if (this.equals(base.Int64.min)) {
            return base.Int64.min;
        }
        return this.not().add(base.Int64.one);
    }

    not() {
        return new Int64(~this.low, ~this.high);
    }

    equals(other) {
        if (!(other instanceof base.Int64) && (this.high >>> 31) === 1 && (other.high >>> 31) === 1) {
            return false;
        }
        return this.high === other.high && this.low === other.low;
    }

    compare(other) {
        if (this.equals(other)) {
            return 0;
        }
        const thisNeg = this.isNegative;
        const otherNeg = other.isNegative;
        if (thisNeg && !otherNeg) {
            return -1;
        }
        if (!thisNeg && otherNeg) {
            return 1;
        }
        return this.subtract(other).isNegative ? -1 : 1;
    }

    add(other) {
        return base.Utility.add(this, other, false);
    }

    subtract(other) {
        return base.Utility.subtract(this, other, false);
    }

    multiply(other) {
        return base.Utility.multiply(this, other, false);
    }

    divide(other) {
        return base.Utility.divide(this, other, false);
    }

    toInteger() {
        return this.low;
    }

    toNumber() {
        if (this.high === 0) {
            return this.low >>> 0;
        }
        if (this.high === -1) {
            return this.low;
        }
        return (this.high * 4294967296) + (this.low >>> 0);
    }

    toString(radix) {
        const r = radix || 10;
        if (r < 2 || r > 16) {
            throw RangeError('radix');
        }
        if (this.isZero) {
            return '0';
        }
        if (this.high < 0) {
            if (this.equals(base.Int64.min)) {
                const r = new Int64(radix, 0);
                const div = this.divide(r);
                const remainder = div.multiply(r).subtract(this);
                return div.toString(r) + (remainder.low >>> 0).toString(r);
            }
            return '-' + this.negate().toString(r);
        }
        if (this.high === 0) {
            return this.low.toString(radix);
        }
        return base.Utility.text(this, false, r);
    }
};

base.Int64.min = new base.Int64(0, -2147483648);
base.Int64.zero = new base.Int64(0, 0);
base.Int64.one = new base.Int64(1, 0);
base.Int64.power24 = new base.Int64(1 << 24, 0);
base.Int64.max = new base.Int64(0, 2147483647);

base.Uint64 = class Uint64 {

    constructor(low, high) {
        this.low = low | 0;
        this.high = high | 0;
    }

    static create(value) {
        if (isNaN(value)) {
            return base.Uint64.zero;
        }
        if (value < 0) {
            return base.Uint64.zero;
        }
        if (value >= 18446744073709552000) {
            return base.Uint64.max;
        }
        if (value < 0) {
            return base.Uint64.create(-value).negate();
        }
        return new base.Uint64((value % 4294967296) | 0, (value / 4294967296));
    }

    get isZero() {
        return this.low === 0 && this.high === 0;
    }

    get isNegative() {
        return false;
    }

    negate() {
        return this.not().add(base.Int64.one);
    }

    not() {
        return new base.Uint64(~this.low, ~this.high);
    }

    equals(other) {
        if (!(other instanceof base.Uint64) && (this.high >>> 31) === 1 && (other.high >>> 31) === 1) {
            return false;
        }
        return this.high === other.high && this.low === other.low;
    }

    compare(other) {
        if (this.equals(other)) {
            return 0;
        }
        const thisNeg = this.isNegative;
        const otherNeg = other.isNegative;
        if (thisNeg && !otherNeg) {
            return -1;
        }
        if (!thisNeg && otherNeg) {
            return 1;
        }
        return (other.high >>> 0) > (this.high >>> 0) || (other.high === this.high && (other.low >>> 0) > (this.low >>> 0)) ? -1 : 1;
    }

    add(other) {
        return base.Utility.add(this, other, true);
    }

    subtract(other) {
        return base.Utility.subtract(this, other, true);
    }

    multiply(other) {
        return base.Utility.multiply(this, other, true);
    }

    divide(other) {
        return base.Utility.divide(this, other, true);
    }

    toInteger() {
        return this.low >>> 0;
    }

    toNumber() {
        if (this.high === 0) {
            return this.low >>> 0;
        }
        return ((this.high >>> 0) * 4294967296) + (this.low >>> 0);
    }

    toString(radix) {
        const r = radix || 10;
        if (r < 2 || 36 < r) {
            throw RangeError('radix');
        }
        if (this.isZero) {
            return '0';
        }
        if (this.high === 0) {
            return this.low.toString(radix);
        }
        return base.Utility.text(this, true, r);
    }
};

base.Utility = class {

    static add(a, b, unsigned) {
        const a48 = a.high >>> 16;
        const a32 = a.high & 0xFFFF;
        const a16 = a.low >>> 16;
        const a00 = a.low & 0xFFFF;
        const b48 = b.high >>> 16;
        const b32 = b.high & 0xFFFF;
        const b16 = b.low >>> 16;
        const b00 = b.low & 0xFFFF;
        let c48 = 0;
        let c32 = 0;
        let c16 = 0;
        let c00 = 0;
        c00 += a00 + b00;
        c16 += c00 >>> 16;
        c00 &= 0xFFFF;
        c16 += a16 + b16;
        c32 += c16 >>> 16;
        c16 &= 0xFFFF;
        c32 += a32 + b32;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c48 += a48 + b48;
        c48 &= 0xFFFF;
        return base.Utility._create((c16 << 16) | c00, (c48 << 16) | c32, unsigned);
    }

    static subtract(a, b, unsigned) {
        return base.Utility.add(a, b.negate(), unsigned);
    }

    static multiply(a, b, unsigned) {
        if (a.isZero) {
            return base.Int64.zero;
        }
        if (b.isZero) {
            return base.Int64.zero;
        }
        if (a.equals(base.Int64.min)) {
            return b.isOdd() ? base.Int64.min : base.Int64.zero;
        }
        if (b.equals(base.Int64.min)) {
            return b.isOdd() ? base.Int64.min : base.Int64.zero;
        }
        if (a.isNegative) {
            if (b.isNegative) {
                return this.negate().multiply(b.negate());
            }
            else {
                return this.negate().multiply(b).negate();
            }
        }
        else if (b.isNegative) {
            return this.multiply(b.negate()).negate();
        }
        if (a.compare(base.Int64.power24) < 0 && b.compare(base.Int64.power24) < 0) {
            return unsigned ? base.Uint64.create(a.toNumber() * b.toNumber()) : base.Int64.create(a.toNumber() * b.toNumber());
        }
        const a48 = a.high >>> 16;
        const a32 = a.high & 0xFFFF;
        const a16 = a.low >>> 16;
        const a00 = a.low & 0xFFFF;
        const b48 = b.high >>> 16;
        const b32 = b.high & 0xFFFF;
        const b16 = b.low >>> 16;
        const b00 = b.low & 0xFFFF;
        let c48 = 0;
        let c32 = 0;
        let c16 = 0;
        let c00 = 0;
        c00 += a00 * b00;
        c16 += c00 >>> 16;
        c00 &= 0xFFFF;
        c16 += a16 * b00;
        c32 += c16 >>> 16;
        c16 &= 0xFFFF;
        c16 += a00 * b16;
        c32 += c16 >>> 16;
        c16 &= 0xFFFF;
        c32 += a32 * b00;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c32 += a16 * b16;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c32 += a00 * b32;
        c48 += c32 >>> 16;
        c32 &= 0xFFFF;
        c48 += a48 * b00 + a32 * b16 + a16 * b32 + a00 * b48;
        c48 &= 0xFFFF;
        return base.Utility._create((c16 << 16) | c00, (c48 << 16) | c32, unsigned);
    }

    static divide(a, b, unsigned) {
        if (b.isZero) {
            throw Error('Division by zero.');
        }
        if (a.isZero) {
            return unsigned ? base.Uint64.zero : base.Int64.zero;
        }
        let approx;
        let remainder;
        let result;
        if (!unsigned) {
            if (a.equals(base.Int64.min)) {
                if (b.equals(base.Int64.one) || b.equals(base.Int64.negativeOne)) {
                    return base.Int64.min;
                }
                else if (b.equals(base.Int64.min)) {
                    return base.Int64.one;
                }
                else {
                    const half = base.Utility._shiftRight(a, unsigned, 1);
                    const halfDivide = half.divide(b);
                    approx = base.Utility._shiftLeft(halfDivide, halfDivide instanceof base.Uint64, 1);
                    if (approx.eq(base.Int64.zero)) {
                        return b.isNegative ? base.Int64.one : base.Int64.negativeOne;
                    }
                    else {
                        remainder = a.subtract(b.multiply(approx));
                        result = approx.add(remainder.divide(b));
                        return result;
                    }
                }
            }
            else if (b.equals(base.Int64.min)) {
                return unsigned ? base.Uint64.zero : base.Int64.zero;
            }
            if (a.isNegative) {
                if (b.isNegative) {
                    return this.negate().divide(b.negate());
                }
                return a.negate().divide(b).negate();
            }
            else if (b.isNegative) {
                return a.divide(b.negate()).negate();
            }
            result = base.Int64.zero;
        }
        else {
            if (!(b instanceof base.Uint64)) {
                b = new base.Uint64(b.low, b.high);
            }
            if (b.compare(a) > 0) {
                return base.Int64.zero;
            }
            if (b.compare(base.Utility._shiftRight(a, unsigned, 1)) > 0) {
                return base.Uint64.one;
            }
            result = base.Uint64.zero;
        }
        remainder = a;
        while (remainder.compare(b) >= 0) {
            let approx = Math.max(1, Math.floor(remainder.toNumber() / b.toNumber()));
            const log2 = Math.ceil(Math.log(approx) / Math.LN2);
            const delta = (log2 <= 48) ? 1 : Math.pow(2, log2 - 48);
            let approxResult = base.Int64.create(approx);
            let approxRemainder = approxResult.multiply(b);
            while (approxRemainder.isNegative || approxRemainder.compare(remainder) > 0) {
                approx -= delta;
                approxResult = unsigned ? base.Uint64.create(approx) : base.Int64.create(approx);
                approxRemainder = approxResult.multiply(b);
            }
            if (approxResult.isZero) {
                approxResult = base.Int64.one;
            }
            result = result.add(approxResult);
            remainder = remainder.subtract(approxRemainder);
        }
        return result;
    }

    static text(value, unsigned, radix) {
        const power = unsigned ? base.Uint64.create(Math.pow(radix, 6)) : base.Int64.create(Math.pow(radix, 6));
        let remainder = value;
        let result = '';
        for (;;) {
            const remainderDiv = remainder.divide(power);
            const intval = remainder.subtract(remainderDiv.multiply(power)).toInteger() >>> 0;
            let digits = intval.toString(radix);
            remainder = remainderDiv;
            if (remainder.low === 0 && remainder.high === 0) {
                return digits + result;
            }
            while (digits.length < 6) {
                digits = '0' + digits;
            }
            result = '' + digits + result;
        }
    }

    static _shiftLeft(value, unsigned, shift) {
        return base.Utility._create(value.low << shift, (value.high << shift) | (value.low >>> (32 - shift)), unsigned);
    }

    static _shiftRight(value, unsigned, shift) {
        return base.Utility._create((value.low >>> shift) | (value.high << (32 - shift)), value.high >> shift, unsigned);
    }

    static _create(low, high, unsigned) {
        return unsigned ? new base.Uint64(low, high) : new base.Int64(low, high);
    }
};

base.Uint64.zero = new base.Uint64(0, 0);
base.Uint64.one = new base.Uint64(1, 0);
base.Uint64.max = new base.Uint64(-1, -1);

if (!DataView.prototype.getFloat16) {
    DataView.prototype.getFloat16 = function(byteOffset, littleEndian) {
        const value = this.getUint16(byteOffset, littleEndian);
        const e = (value & 0x7C00) >> 10;
        let f = value & 0x03FF;
        if (e == 0) {
            f = 0.00006103515625 * (f / 1024);
        }
        else if (e == 0x1F) {
            f = f ? NaN : Infinity;
        }
        else {
            f = DataView.__float16_pow[e] * (1 + (f / 1024));
        }
        return value & 0x8000 ? -f : f;
    };
    DataView.__float16_pow = {
        1: 1/16384, 2: 1/8192, 3: 1/4096, 4: 1/2048, 5: 1/1024, 6: 1/512, 7: 1/256, 8: 1/128,
        9: 1/64, 10: 1/32, 11: 1/16, 12: 1/8, 13: 1/4, 14: 1/2, 15: 1, 16: 2,
        17: 4, 18: 8, 19: 16, 20: 32, 21: 64, 22: 128, 23: 256, 24: 512,
        25: 1024, 26: 2048, 27: 4096, 28: 8192, 29: 16384, 30: 32768, 31: 65536
    };
}

if (!DataView.prototype.setFloat16) {
    DataView.prototype.setFloat16 = function(byteOffset, value, littleEndian) {
        DataView.__float16_float[0] = value;
        value = DataView.__float16_int[0];
        const s = (value >>> 16) & 0x8000;
        const e = (value >>> 23) & 0xff;
        const f = value & 0x7fffff;
        const v = s | DataView.__float16_base[e] | (f >> DataView.__float16_shift[e]);
        this.setUint16(byteOffset, v, littleEndian);
    };
    DataView.__float16_float = new Float32Array(1);
    DataView.__float16_int = new Uint32Array(DataView.__float16_float.buffer, 0, DataView.__float16_float.length);
    DataView.__float16_base = new Uint32Array(256);
    DataView.__float16_shift = new Uint32Array(256);
    for (let i = 0; i < 256; ++i) {
        const e = i - 127;
        if (e < -27) {
            DataView.__float16_base[i] = 0x0000;
            DataView.__float16_shift[i] = 24;
        }
        else if (e < -14) {
            DataView.__float16_base[i] = 0x0400 >> -e - 14;
            DataView.__float16_shift[i] = -e - 1;
        }
        else if (e <= 15) {
            DataView.__float16_base[i] = e + 15 << 10;
            DataView.__float16_shift[i] = 13;
        }
        else if (e < 128) {
            DataView.__float16_base[i] = 0x7c00;
            DataView.__float16_shift[i] = 24;
        }
        else {
            DataView.__float16_base[i] = 0x7c00;
            DataView.__float16_shift[i] = 13;
        }
    }
}

DataView.prototype.getInt64 = DataView.prototype.getInt64 || function(byteOffset, littleEndian) {
    return littleEndian ?
        new base.Int64(this.getUint32(byteOffset, true), this.getUint32(byteOffset + 4, true)) :
        new base.Int64(this.getUint32(byteOffset + 4, true), this.getUint32(byteOffset, true));
};

DataView.prototype.setInt64 = DataView.prototype.setInt64 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setUint32(byteOffset, value.low, true);
        this.setUint32(byteOffset + 4, value.high, true);
    }
    else {
        this.setUint32(byteOffset + 4, value.low, false);
        this.setUint32(byteOffset, value.high, false);
    }
};

DataView.prototype.getUint64 = DataView.prototype.getUint64 || function(byteOffset, littleEndian) {
    return littleEndian ?
        new base.Uint64(this.getUint32(byteOffset, true), this.getUint32(byteOffset + 4, true)) :
        new base.Uint64(this.getUint32(byteOffset + 4, true), this.getUint32(byteOffset, true));
};

DataView.prototype.setUint64 = DataView.prototype.setUint64 || function(byteOffset, value, littleEndian) {
    if (littleEndian) {
        this.setUInt32(byteOffset, value.low, true);
        this.setUInt32(byteOffset + 4, value.high, true);
    }
    else {
        this.setUInt32(byteOffset + 4, value.low, false);
        this.setUInt32(byteOffset, value.high, false);
    }
};

DataView.prototype.getBits = DataView.prototype.getBits || function(offset, bits /*, signed */) {
    offset = offset * bits;
    const available = (this.byteLength << 3) - offset;
    if (bits > available) {
        throw new RangeError();
    }
    let value = 0;
    let index = 0;
    while (index < bits) {
        const remainder = offset & 7;
        const size = Math.min(bits - index, 8 - remainder);
        value <<= size;
        value |= (this.getUint8(offset >> 3) >> (8 - size - remainder)) & ~(0xff << size);
        offset += size;
        index += size;
    }
    return value;
};

base.BinaryReader = class {

    constructor(data) {
        this._buffer = data instanceof Uint8Array ? data : data.peek();
        this._position = 0;
        this._length = this._buffer.length;
        this._view = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        this._utf8 = new TextDecoder('utf-8');
    }

    get length() {
        return this._length;
    }

    get position() {
        return this._position;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._length || this._position < 0) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    read(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        return this._buffer.slice(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }

    int8() {
        const position = this._position;
        this.skip(1);
        return this._view.getInt8(position, true);
    }

    int16() {
        const position = this._position;
        this.skip(2);
        return this._view.getInt16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._view.getInt32(position, true);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._view.getInt64(position, true).toNumber();
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, true);
    }

    uint64() {
        const position = this._position;
        this.skip(8);
        return this._view.getUint64(position, true).toNumber();
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._view.getFloat32(position, true);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._view.getFloat64(position, true);
    }

    string() {
        const length = this.uint32();
        const position = this._position;
        this.skip(length);
        const data = this._buffer.subarray(position, this._position);
        return this._utf8.decode(data);
    }
};

if (typeof window !== 'undefined' && typeof window.Long != 'undefined') {
    window.long = { Long: window.Long };
    window.Int64 = base.Int64;
    window.Uint64 = base.Uint64;
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Int64 = base.Int64;
    module.exports.Uint64 = base.Uint64;
    module.exports.BinaryReader = base.BinaryReader;
}
