
async function testInference() {
    const session = await ort.InferenceSession.create('C:/Users/edmun/Desktop/VSCode Projects/TeXan/model_2.onnx');
    console.log("Model loading successful");
    
    const src_array = new Float32Array(3*384*512); 
    const src_tensor = new ort.Tensor( 'float32', src_array, [1, 3, 384, 512] );
    
    const tgt_array = new BigInt64Array(2);
    const tgt_tensor = new ort.Tensor('int64', tgt_array, [1, 2] );
    
    const tgt_mask_array = new Float32Array(4);
    const tgt_mask_tensor = new ort.Tensor( 'float32', tgt_mask_array, [2, 2] );

    const test_output = await session.run(
        {src: src_tensor, tgt: tgt_tensor, tgt_mask: tgt_mask_tensor}
    );

    console.log("test output: ", test_output)

}



async function loadLocalModel() {
    // Get local URLs for the ONNX files
    const modelUrl = chrome.runtime.getURL('model_2.onnx');
    const dataUrl = chrome.runtime.getURL('model_2.onnx.data');
  
    // Fetch the files
    const responseModel = await fetch(modelUrl);
    const modelBlob = await responseModel.blob();
  
    const responseData = await fetch(dataUrl);
    const dataBlob = await responseData.blob();
  
    // Convert blobs to ArrayBuffers
    const modelBuffer = await modelBlob.arrayBuffer();
    const dataBuffer = await dataBlob.arrayBuffer();
  
    // Load the ONNX model
    const mySession = await ort.InferenceSession.create(modelBlob, {
      externalData: [
        {
          path: './model_2.onnx.data',
          data: dataBlob
        }
      ]
    });
  
    console.log('Local ONNX model loaded successfully:', mySession);
  }
  
