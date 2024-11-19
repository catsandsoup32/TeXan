

class LaTeX_Tokenizer { 
    constructor() { 
        this.id_to_token = {0: '[PAD]', 1: '[BOS]', 2: '[EOS]', 3: '[UNK]', 4: '{', 5: '}', 6: '_', 7: 'x', 8: '^', 9: ')', 10: '(', 11: '=', 12: '1', 13: 'i', 14: 't', 15: '2', 16: 'a', 17: 'r', 18: 'm', 19: '-', 20: '\\frac', 21: '0', 22: 'd', 23: 'n', 24: ',', 25: '+', 26: '\\\\', 27: '\\begin', 28: '\\end', 29: '|', 30: 'k', 31: '&', 32: 'f', 33: '\\int', 34: '\\sqrt', 35: '3', 36: 'y', 37: 'p', 38: '\\hat', 39: 'A', 40: 's', 41: ']', 42: '[', 43: '\\partial', 44: 'c', 45: 'e', 46: '\\tilde', 47: '.', 48: '/', 49: '4', 50: 'X', 51: 'b', 52: 'j', 53: 'P', 54: 'v', 55: 'C', 56: 'S', 57: 'g', 58: 'u', 59: 'R', 60: 'z', 61: 'T', 62: 'l', 63: '\\prime', 64: 'E', 65: 'N', 66: '\\overline', 67: 'F', 68: 'B', 69: 'L', 70: 'V', 71: '5', 72: 'o', 73: '\\mu', 74: 'q', 75: 'I', 76: '\\cdot', 77: 'M', 78: '\\alpha', 79: '\\pi', 80: 'H', 81: 'D', 82: '\\}', 83: '\\{', 84: '6', 85: 'h', 86: '\\in', 87: 'G', 88: '\\sum', 89: '\\lambda', 90: 'K', 91: '*', 92: '\\prod', 93: '<', 94: 'w', 95: '\\theta', 96: 'Q', 97: '\\sigma', 98: ':', 99: '\\infty', 100: 'U', 101: '\\omega', 102: 'Y', 103: '\\rho', 104: 'Z', 105: '\\rangle', 106: '\\beta', 107: '7', 108: '\\rightarrow', 109: '\\gamma', 110: '\\epsilon', 111: 'O', 112: '\\underline', 113: '\\phi', 114: '\\le', 115: '\\notin', 116: '\\varphi', 117: 'W', 118: '\\delta', 119: '\\psi', 120: '8', 121: '\\nu', 122: '>', 123: '\\vec', 124: '\\langle', 125: '\\Delta', 126: 'J', 127: '\\times', 128: '\\dot', 129: '\\Omega', 130: '!', 131: '\\tau', 132: '9', 133: '\\pm', 134: '\\chi', 135: '\\approx', 136: '\\eta', 137: ';', 138: '\\nabla', 139: '\\mathbb', 140: '\\xi', 141: '\\Phi', 142: '\\ge', 143: '\\Psi', 144: '\\Sigma', 145: '\\sim', 146: '\\zeta', 147: '\\circ', 148: '\\Gamma', 149: '\\ne', 150: '\\forall', 151: '\\Lambda', 152: '\\mapsto', 153: '\\otimes', 154: '\\hbar', 155: '\\cup', 156: '\\equiv', 157: '\\kappa', 158: '\\Pi', 159: '\\oplus', 160: '\\subset', 161: '\\cap', 162: '\\bigcup', 163: '\\subseteq', 164: '\\wedge', 165: '\\cong', 166: '\\neg', 167: '\\Theta', 168: '\\dagger', 169: '\\oint', 170: '\\Rightarrow', 171: '\\aleph', 172: '\\lfloor', 173: '\\rfloor', 174: '\\backslash', 175: '\\emptyset', 176: '\\perp', 177: '\\#', 178: '\\propto', 179: '\\%', 180: '\\simeq', 181: '\\vee', 182: '?', 183: '\\ll', 184: '\\Vdash', 185: '\\Xi', 186: '\\leftarrow', 187: '\\bigcap', 188: '\\longrightarrow', 189: '\\bullet', 190: '\\exists', 191: '\\iint', 192: '\\vdash', 193: '\\iff', 194: '\\top', 195: '\\|', 196: '\\bigoplus', 197: '\\odot', 198: '\\lceil', 199: '\\rceil', 200: '\\leftrightarrow', 201: '\\models', 202: '\\supseteq', 203: '\\bigwedge', 204: '\\varsigma', 205: '\\rightleftharpoons', 206: '\\angle', 207: '\\vdots', 208: '\\Leftrightarrow', 209: '\\subsetneq', 210: '\\iota', 211: '\\gg', 212: '\\ominus', 213: '\\supset', 214: '\\Upsilon', 215: '\\triangle', 216: '\\_'}
        this.id_to_token_map = new Map();
        for (let i=0; i<Object.keys(this.id_to_token).length; i++) {
            this.id_to_token_map.set(i, this.id_to_token[String(i)]);
        }
    } 

    decode(input_id_list) {
        for (let i=0; i<input_id_list.length; i++) {
           input_id_list[i] = this.id_to_token_map.get(input_id_list[i])
        }
        return input_id_list;
    }

}

function generate_image_tensor(image_data) {
    const image_data_array = image_data.data; // 1-D array of RGBA, order by rows top left to bottom right
    let processed_image_data = new Float32Array(589824); // [3 * 512 * 384] to match dimensions expected by model

    for (let i=0, j=0; i < 196608; i++, j += 4) {
        processed_image_data[i] = image_data_array[j] / 255; // Red
        processed_image_data[i + 196608] = image_data_array[j + 1] / 255; // Green
        processed_image_data[i + 393216] = image_data_array[j + 2] / 255; // Blue
    }
    
    const image_tensor = new ort.Tensor('float32', processed_image_data, [1, 3, 384, 512]);
    console.log("Image tensor ", image_tensor.data);
    return image_tensor;
}

function generate_tgt_in(tgt_in_values_slice) {
    const tgt_in_tensor = new ort.Tensor('int64', BigInt64Array.from(tgt_in_values_slice), [1, tgt_in_values_slice.length]);
    console.log("Tgt_in tensor ", tgt_in_tensor);
    return tgt_in_tensor;
}

function generate_tgt_mask(size) { 
    // Returns a square matrix with everything on and above diagonal set to negative infinity ( softmax(-infinity) = 0 )
    let mask_array = new Float32Array(size*size);
    let counter = 1;
    let offset;

    for (let i=0; i < size*size; i++) {
        offset = size*i;
        mask_array.fill(0, offset, counter + offset);
        mask_array.fill(Number.NEGATIVE_INFINITY, counter + offset, offset + size); 
        counter++;
    }

    const tgt_mask_tensor = new ort.Tensor('float32', mask_array, [size, size]);
    console.log("Tgt_mask tensor ", tgt_mask_tensor);
    return tgt_mask_tensor;
}



