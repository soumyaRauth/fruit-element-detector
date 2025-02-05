"use client";
import { useState } from 'react';
import Image from "next/image";

interface NutrientInfo {
  amount: string;
  description: string;
}

interface BananaAnalysis {
  characteristics: {
    ripeness: string;
    color: string;
    spots: string;
    size: string;
  };
  nutrients: {
    [key: string]: NutrientInfo;
  };
  confidence: number;
  analysis_time: string;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<BananaAnalysis | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!selectedFile) return;

    setLoading(true);
    setAnalysis(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image');
      }

      const result = await response.json();
      setAnalysis(result);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to analyze image.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-gray-50">
      <main className="max-w-4xl mx-auto">
        <section className="bg-white p-8 rounded-lg shadow-lg">
          <h1 className="text-3xl font-bold mb-8 text-center">Banana Chemical Analysis</h1>
          
          <form onSubmit={handleSubmit} className="mb-8">
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Banana Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="w-full p-2 border rounded"
              />
            </div>
            <button
              type="submit"
              disabled={!selectedFile || loading}
              className={`w-full p-3 rounded-md font-medium ${
                !selectedFile || loading
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              {loading ? 'Analyzing...' : 'Analyze Banana'}
            </button>
          </form>

          {analysis && (
            <div className="space-y-6">
              <div className="bg-blue-50 p-4 rounded-md">
                <h2 className="text-xl font-semibold mb-2">Banana Characteristics</h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Ripeness</p>
                    <p className="font-medium capitalize">{analysis.characteristics.ripeness}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Color</p>
                    <p className="font-medium capitalize">{analysis.characteristics.color}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Spots</p>
                    <p className="font-medium capitalize">{analysis.characteristics.spots}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Size</p>
                    <p className="font-medium capitalize">{analysis.characteristics.size}</p>
                  </div>
                </div>
              </div>

              <div>
                <h2 className="text-xl font-semibold mb-4">Chemical Composition</h2>
                <div className="grid gap-4 md:grid-cols-2">
                  {Object.entries(analysis.nutrients).map(([nutrient, info]) => (
                    <div key={nutrient} className="p-4 border rounded-md">
                      <h3 className="font-medium capitalize mb-1">{nutrient}</h3>
                      <p className="text-2xl font-bold text-blue-600 mb-2">{info.amount}</p>
                      <p className="text-sm text-gray-600">{info.description}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div className="text-sm text-gray-500 border-t pt-4">
                <p>Analysis Confidence: {(analysis.confidence * 100).toFixed(1)}%</p>
                <p>Analyzed at: {new Date(analysis.analysis_time).toLocaleString()}</p>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
