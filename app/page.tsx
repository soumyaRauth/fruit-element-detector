"use client";

import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import TrainingInterface from './components/TrainingInterface';
import PredictionInterface from './components/PredictionInterface';
import { ThemeToggle } from './components/ThemeToggle';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'train' | 'predict'>('train');
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [isModelReady, setIsModelReady] = useState(false);

  useEffect(() => {
    // Initialize TensorFlow.js
    tf.ready().then(() => {
      console.log('TensorFlow.js is ready');
    });
  }, []);

  const handleModelTrained = (trainedModel: tf.LayersModel) => {
    setModel(trainedModel);
    setIsModelReady(true);
    setActiveTab('predict');
  };

  const handleTabChange = (tab: 'train' | 'predict') => {
    if (tab === 'predict' && !isModelReady) {
      alert('Please train the model first');
      return;
    }
    setActiveTab(tab);
  };

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto py-8">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-foreground">
            Fruit Toxic Chemical Detection
          </h1>
          <ThemeToggle />
        </div>

        <div className="bg-card rounded-lg shadow-lg overflow-hidden border">
          <div className="flex border-b border-border">
            <button
              className={`flex-1 py-4 px-6 text-center transition-colors ${
                activeTab === 'train'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-card text-muted-foreground hover:text-foreground hover:bg-accent'
              }`}
              onClick={() => handleTabChange('train')}
            >
              Train Model
            </button>
            <button
              className={`flex-1 py-4 px-6 text-center transition-colors ${
                activeTab === 'predict'
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-card text-muted-foreground hover:text-foreground hover:bg-accent'
              } ${!isModelReady ? 'opacity-50 cursor-not-allowed' : ''}`}
              onClick={() => handleTabChange('predict')}
              disabled={!isModelReady}
            >
              Predict
            </button>
          </div>

          <div className="p-6">
            {activeTab === 'train' ? (
              <TrainingInterface onModelTrained={handleModelTrained} />
            ) : (
              <PredictionInterface model={model} />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
