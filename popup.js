
/*
Simple vanilla js canvas 
*/

const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
ctx.imageSmoothingEnabled = true;
ctx.lineCap = 'round';
ctx.lineWidth = 5;
ctx.strokeStyle = 'black';

// Drawing event listeners

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

document.getElementById('pencil').addEventListener('click', () => { 
    changeColor('black'); });
document.getElementById('eraser').addEventListener('click', () => {
    changeColor('white'); });
document.getElementById('clear').addEventListener('click', () => {
    clearAll(); });

let isDrawing = false; // Track if the user is drawing

// Mouse event functions
function startDrawing(event) {
  isDrawing = true;
  ctx.beginPath();
  ctx.moveTo(event.offsetX, event.offsetY);
}

function draw(event) {
  if (!isDrawing) return; // Stop drawing if the mouse is not down
  ctx.lineTo(event.offsetX, event.offsetY);
  ctx.stroke();
}

function stopDrawing() {
  isDrawing = false;
  ctx.closePath();
}

function changeColor(color) { 
    ctx.strokeStyle = color;
    if (color === 'black') {
        ctx.lineWidth = 5;
    } else {
        ctx.lineWidth = 50;
    }
}

function clearAll() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/*
Helpers for inference
*/ 

const tokenizer = new LaTeX_Tokenizer();

async function loadModel() {
    const onnxFileBuffer = await (await fetch('model_3.onnx')).arrayBuffer()
    const onnxArray = new Uint8Array(onnxFileBuffer);
    const dataFileBuffer = await (await fetch('model_3.onnx.data')).arrayBuffer()
    const dataArray = new Uint8Array(dataFileBuffer)
    const sessionPromise = await ort.InferenceSession.create(onnxArray, {
        executionProviders: ["wasm"] ,
        externalData: [{path: './model_3.onnx.data', data: dataArray}]
    });

    const model = await sessionPromise;
    return model;
}


async function runInference(max_size, model) {
    const image_tensor = generate_image_tensor(ctx.getImageData(0, 0, 512, 384));
    let output;
    let tgt_in_values = new BigInt64Array(max_size);

    for (let i=1; i < max_size; i++) {
        output = await model.run({ 
            'src': image_tensor,
            'tgt': generate_tgt_in(tgt_in_values.slice(0, i)),
            'tgt_mask': generate_tgt_mask(i)
        }); 
        tgt_in_values[i] = tgt_in_values[0];
        console.log("output: ", output);
    }
}

async function main() {
    const model = await loadModel();
    runInference(5, model);
}


main();