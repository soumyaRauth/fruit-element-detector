'use client';

import { useState, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Upload, Loader2, Save } from 'lucide-react';

interface TrainingData {
  image: File;
  label: string;
}

interface TrainingInterfaceProps {
  onModelTrained: (model: tf.LayersModel) => void;
}

export default function TrainingInterface({ onModelTrained }: TrainingInterfaceProps) {
  const [trainingData, setTrainingData] = useState<TrainingData[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<{ epoch: number; loss?: number }>({ epoch: 0 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    
    const files = Array.from(e.target.files);
    const newTrainingData: TrainingData[] = files.map(file => ({
      image: file,
      label: 'unlabeled'
    }));
    
    setTrainingData([...trainingData, ...newTrainingData]);
  };

  const createModel = () => {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
      inputShape: [224, 224, 3],
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    
    model.compile({
      optimizer: 'adam',
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });
    
    return model;
  };

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

  const saveModel = async (model: tf.LayersModel) => {
    try {
      // Save model to IndexedDB for browser persistence
      await model.save('indexeddb://toxic-detection-model');
      
      // Also save model architecture and weights as files
      const saveResult = await model.save('downloads://toxic-detection-model');
      console.log('Model saved:', saveResult);
    } catch (error) {
      console.error('Error saving model:', error);
    }
  };

  const startTraining = async () => {
    setIsTraining(true);
    setProgress({ epoch: 0 });
    const model = createModel();

    try {
      const batchSize = 32;
      const epochs = 10;

      for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;
        let batchCount = 0;

        for (let i = 0; i < trainingData.length; i += batchSize) {
          const batch = trainingData.slice(i, Math.min(i + batchSize, trainingData.length));
          const xs = await Promise.all(batch.map(data => preprocessImage(data.image)));
          const ys = batch.map(data => data.label === 'toxic' ? 1 : 0);

          const xsTensor = tf.concat(xs);
          const ysTensor = tf.tensor2d(ys, [ys.length, 1]);

          const result = await model.trainOnBatch(xsTensor, ysTensor);
          const loss = Array.isArray(result) ? result[0] : result;
          totalLoss += loss;
          batchCount++;

          xsTensor.dispose();
          ysTensor.dispose();
        }

        const averageLoss = totalLoss / batchCount;
        setProgress({ epoch: epoch + 1, loss: averageLoss });
        console.log(`Epoch ${epoch + 1}/${epochs} completed - Loss: ${averageLoss.toFixed(4)}`);
      }
      
      // Save the model before notifying completion
      await saveModel(model);
      onModelTrained(model);
    } catch (error) {
      console.error('Training error:', error);
    }

    setIsTraining(false);
  };

  const setLabel = (index: number, label: string) => {
    const newTrainingData = [...trainingData];
    newTrainingData[index].label = label;
    setTrainingData(newTrainingData);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button
          onClick={() => fileInputRef.current?.click()}
          className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
        >
          <Upload className="h-4 w-4" />
          Upload Images
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
        />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {trainingData.map((data, index) => (
          <div key={index} className="overflow-hidden rounded-lg border bg-card text-card-foreground shadow-sm">
            <div className="aspect-video w-full">
              <img
                src={URL.createObjectURL(data.image)}
                alt={`Training image ${index}`}
                className="h-full w-full object-cover"
              />
            </div>
            <div className="p-4">
              <select
                value={data.label}
                onChange={(e) => setLabel(index, e.target.value)}
                className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                <option value="unlabeled">Select Label</option>
                <option value="toxic">Toxic</option>
                <option value="non-toxic">Non-Toxic</option>
              </select>
            </div>
          </div>
        ))}
      </div>

      {isTraining && progress.loss !== undefined && (
        <div className="text-sm text-muted-foreground">
          Training Progress: Epoch {progress.epoch}/10 - Loss: {progress.loss.toFixed(4)}
        </div>
      )}

      <button
        onClick={startTraining}
        disabled={isTraining || trainingData.length === 0 || trainingData.some(data => data.label === 'unlabeled')}
        className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground ring-offset-background transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
      >
        {isTraining ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Training...
          </>
        ) : (
          'Start Training'
        )}
      </button>
    </div>
  );
} 