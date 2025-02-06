'use client';

import { useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Upload, Loader2, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function PredictionInterface({ model }: { model: tf.LayersModel | null }) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const preprocessImage = async (file: File): Promise<tf.Tensor3D> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const tensor = tf.browser.fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .toFloat()
          .div(255.0)
          .expandDims(0);
        resolve(tensor as tf.Tensor3D);
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    setSelectedImage(e.target.files[0]);
    setPrediction(null);
  };

  const makePrediction = async () => {
    if (!model || !selectedImage) return;

    setIsProcessing(true);
    try {
      const tensor = await preprocessImage(selectedImage);
      const prediction = await model.predict(tensor) as tf.Tensor;
      const probabilityValue = await prediction.data();
      
      setPrediction(probabilityValue[0] > 0.5 ? 'Toxic' : 'Non-Toxic');
      
      tensor.dispose();
      prediction.dispose();
    } catch (error) {
      console.error('Prediction error:', error);
      setPrediction('Error during prediction');
    }
    setIsProcessing(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
        >
          <Upload className="h-4 w-4" />
          Select Image
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="hidden"
        />
      </div>

      {selectedImage && (
        <div className="overflow-hidden rounded-lg border bg-card text-card-foreground shadow-sm">
          <div className="aspect-video w-full max-w-md">
            <img
              src={URL.createObjectURL(selectedImage)}
              alt="Selected image"
              className="h-full w-full object-cover"
            />
          </div>
        </div>
      )}

      <button
        onClick={makePrediction}
        disabled={!model || !selectedImage || isProcessing}
        className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
      >
        {isProcessing ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Processing...
          </>
        ) : (
          'Predict'
        )}
      </button>

      {prediction && (
        <div className={`flex items-center gap-2 rounded-lg border p-4 ${
          prediction === 'Toxic' 
            ? 'border-destructive/50 bg-destructive/10 text-destructive' 
            : 'border-green-500/50 bg-green-500/10 text-green-500'
        }`}>
          {prediction === 'Toxic' ? (
            <AlertTriangle className="h-5 w-5" />
          ) : (
            <CheckCircle2 className="h-5 w-5" />
          )}
          <div>
            <h3 className="font-semibold">Result:</h3>
            <p className="text-lg">{prediction}</p>
          </div>
        </div>
      )}
    </div>
  );
} 