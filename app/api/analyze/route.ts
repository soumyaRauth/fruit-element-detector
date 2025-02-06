import { NextResponse } from 'next/server';
import * as tf from '@tensorflow/tfjs-node';

let model: tf.LayersModel | null = null;

async function loadModel() {
  try {
    // Load the model from the saved path
    // We'll need to implement model saving in the training interface
    model = await tf.loadLayersModel('file://./model/model.json');
    return true;
  } catch (error) {
    console.error('Failed to load model:', error);
    return false;
  }
}

async function preprocessImage(imageBuffer: Buffer): Promise<tf.Tensor3D> {
  // Convert buffer to tensor
  const tensor = tf.node.decodeImage(imageBuffer, 3);
  
  // Resize to match our training input size
  const resized = tf.image.resizeBilinear(tensor as tf.Tensor3D, [224, 224]);
  
  // Normalize pixel values
  const normalized = resized.toFloat().div(255.0);
  
  // Cleanup
  tensor.dispose();
  resized.dispose();
  
  return normalized as tf.Tensor3D;
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file');
    
    if (!file || typeof file !== 'object' || !('arrayBuffer' in file)) {
      return NextResponse.json({ error: 'No file uploaded.' }, { status: 400 });
    }

    // Load model if not already loaded
    if (!model) {
      const loaded = await loadModel();
      if (!loaded) {
        return NextResponse.json({ 
          error: 'Model not available. Please train the model first.' 
        }, { status: 503 });
      }
    }

    // Convert File to buffer
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // Preprocess image
    const inputTensor = await preprocessImage(buffer);
    // Make prediction
    const prediction = await model!.predict(inputTensor.expandDims(0)) as tf.Tensor;
    const probability = (await prediction.data())[0];
    
    // Cleanup
    inputTensor.dispose();
    prediction.dispose();

    // Prepare detailed analysis
    const analysis = {
      result: probability > 0.5 ? 'toxic' : 'non-toxic',
      confidence: probability > 0.5 ? probability : 1 - probability,
      details: {
        toxicProbability: probability,
        safetyLevel: probability > 0.8 ? 'High Risk' : 
                     probability > 0.5 ? 'Moderate Risk' : 
                     probability < 0.2 ? 'Very Safe' : 'Safe',
        recommendation: probability > 0.5 
          ? 'This fruit may contain toxic chemicals. Not recommended for consumption.'
          : 'This fruit appears safe for consumption.',
      },
      analysis_time: new Date().toISOString()
    };

    return NextResponse.json(analysis);
  } catch (error) {
    console.error('Failed to analyze image:', error);
    return NextResponse.json({ 
      error: 'Failed to process image.',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
} 