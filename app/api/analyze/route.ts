import { NextResponse } from 'next/server';

// Mock function to analyze banana characteristics
function analyzeBananaImage(imageData: string) {
  // Simulate AI detection of banana characteristics
  const mockAnalysis = {
    ripeness: Math.random() > 0.5 ? 'ripe' : 'slightly-ripe',
    color: Math.random() > 0.5 ? 'yellow' : 'yellow-brown',
    spots: Math.random() > 0.7 ? 'many' : 'few',
    size: Math.random() > 0.5 ? 'large' : 'medium',
  };

  // Adjust chemical composition based on detected characteristics
  const chemicalComposition = {
    potassium: {
      amount: mockAnalysis.ripeness === 'ripe' ? '450mg' : '400mg',
      description: 'Essential mineral for heart and muscle function'
    },
    magnesium: {
      amount: mockAnalysis.size === 'large' ? '34mg' : '28mg',
      description: 'Supports bone health and energy production'
    },
    vitaminC: {
      amount: mockAnalysis.ripeness === 'ripe' ? '12mg' : '14mg',
      description: 'Antioxidant supporting immune system'
    },
    vitaminB6: {
      amount: mockAnalysis.color === 'yellow-brown' ? '0.5mg' : '0.4mg',
      description: 'Vital for brain development and function'
    },
    fiber: {
      amount: mockAnalysis.ripeness === 'ripe' ? '3.1g' : '2.6g',
      description: 'Supports digestive health'
    },
    sugar: {
      amount: mockAnalysis.ripeness === 'ripe' ? '14g' : '12g',
      description: 'Natural fruit sugar content'
    }
  };

  return {
    characteristics: mockAnalysis,
    nutrients: chemicalComposition,
    confidence: 0.92,
    analysis_time: new Date().toISOString()
  };
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    console.log('Received form data');
    
    const file = formData.get('file');
    console.log('File received:', file ? 'yes' : 'no');
    
    if (!file || typeof file !== 'object' || !('arrayBuffer' in file)) {
      console.log('Invalid file:', file);
      return NextResponse.json({ error: 'No file uploaded.' }, { status: 400 });
    }

    // Convert File to base64 (simulating image processing)
    const bytes = await file.arrayBuffer();
    console.log('File converted to array buffer');
    
    const buffer = Buffer.from(bytes);
    console.log('Buffer created');
    
    const base64Image = buffer.toString('base64');
    console.log('Converted to base64');

    // Analyze the image using our mock function
    const analysis = analyzeBananaImage(base64Image);
    console.log('Analysis completed');

    return NextResponse.json(analysis);
  } catch (error) {
    // Log the full error details
    console.error('Failed to analyze image. Error:', error);
    console.error('Error stack:', error instanceof Error ? error.stack : 'No stack trace');
    return NextResponse.json({ 
      error: 'Failed to process image.',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
} 