import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const BenchmarkCharts = () => {
  // Prepare data for each chart
  const scoreData = [
    {
      benchmark: 'Coding',
      'no-streaming-original': 5.65,
      'streaming-original': 7.35,
      'streaming-rag': 7.65
    },
    {
      benchmark: 'Extraction',
      'no-streaming-original': 4.55,
      'streaming-original': 6.95,
      'streaming-rag': 5.75
    },
    {
      benchmark: 'Humanities',
      'no-streaming-original': 3.25,
      'streaming-original': 8.55,
      'streaming-rag': 7.25
    },
    {
      benchmark: 'Math',
      'no-streaming-original': 4.0,
      'streaming-original': 4.65,
      'streaming-rag': 6.35
    },
    {
      benchmark: 'Reasoning',
      'no-streaming-original': 4.75,
      'streaming-original': 8.90,
      'streaming-rag': 8.05
    },
    {
      benchmark: 'Roleplay',
      'no-streaming-original': 4.35,
      'streaming-original': 7.40,
      'streaming-rag': 7.85
    },
    {
      benchmark: 'STEM',
      'no-streaming-original': 3.15,
      'streaming-original': 7.20,
      'streaming-rag': 6.55
    },
    {
      benchmark: 'Writing',
      'no-streaming-original': 5.25,
      'streaming-original': 8.55,
      'streaming-rag': 8.55
    }
  ];

  const generationTimeData = [
    {
      benchmark: 'Coding',
      'no-streaming-original': 2.95,
      'streaming-original': 23.85,
      'streaming-rag': 72.29
    },
    {
      benchmark: 'Extraction',
      'no-streaming-original': 0.34,
      'streaming-original': 1.03,
      'streaming-rag': 6.0
    },
    {
      benchmark: 'Humanities',
      'no-streaming-original': 0.34,
      'streaming-original': 27.10,
      'streaming-rag': 65.74
    },
    {
      benchmark: 'Math',
      'no-streaming-original': 1.62,
      'streaming-original': 15.87,
      'streaming-rag': 44.18
    },
    {
      benchmark: 'Reasoning',
      'no-streaming-original': 0.87,
      'streaming-original': 5.5,
      'streaming-rag': 6.68
    },
    {
      benchmark: 'Roleplay',
      'no-streaming-original': 6.81,
      'streaming-original': 16.78,
      'streaming-rag': 58.31
    },
    {
      benchmark: 'STEM',
      'no-streaming-original': 0.47,
      'streaming-original': 24.03,
      'streaming-rag': 78.18
    },
    {
      benchmark: 'Writing',
      'no-streaming-original': 0.22,
      'streaming-original': 21.83,
      'streaming-rag': 78.74
    }
  ];

  const tokensPerSecondData = [
    {
      benchmark: 'Coding',
      'no-streaming-original': 9.76,
      'streaming-original': 0.95,
      'streaming-rag': 1.32
    },
    {
      benchmark: 'Extraction',
      'no-streaming-original': 10.52,
      'streaming-original': 14.46,
      'streaming-rag': 38.10
    },
    {
      benchmark: 'Humanities',
      'no-streaming-original': 0.14,
      'streaming-original': 1.91,
      'streaming-rag': 1.34
    },
    {
      benchmark: 'Math',
      'no-streaming-original': 65.29,
      'streaming-original': 5.10,
      'streaming-rag': 2.33
    },
    {
      benchmark: 'Reasoning',
      'no-streaming-original': 121.11,
      'streaming-original': 3.10,
      'streaming-rag': 1.74
    },
    {
      benchmark: 'Roleplay',
      'no-streaming-original': 91.96,
      'streaming-original': 10.45,
      'streaming-rag': 12.48
    },
    {
      benchmark: 'STEM',
      'no-streaming-original': 143.53,
      'streaming-original': 2.47,
      'streaming-rag': 2.28
    },
    {
      benchmark: 'Writing',
      'no-streaming-original': 2.41,
      'streaming-original': 3.48,
      'streaming-rag': 0.76
    }
  ];

  const chartProps = {
    width: '100%',
    height: 400,
    margin: { top: 20, right: 30, left: 20, bottom: 60 }
  };

  return (
    <div className="p-4 space-y-8">
      <div>
        <h2 className="text-xl font-bold mb-4 text-center">Benchmark Scores</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={scoreData} {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="benchmark" 
              angle={-45} 
              textAnchor="end" 
              interval={0} 
              height={80}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="no-streaming-original" fill="#8884d8" />
            <Bar dataKey="streaming-original" fill="#82ca9d" />
            <Bar dataKey="streaming-rag" fill="#ffc658" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h2 className="text-xl font-bold mb-4 text-center">Average Generation Time (seconds)</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={generationTimeData} {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="benchmark" 
              angle={-45} 
              textAnchor="end" 
              interval={0} 
              height={80}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="no-streaming-original" fill="#8884d8" />
            <Bar dataKey="streaming-original" fill="#82ca9d" />
            <Bar dataKey="streaming-rag" fill="#ffc658" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h2 className="text-xl font-bold mb-4 text-center">Average Tokens per Second</h2>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={tokensPerSecondData} {...chartProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="benchmark" 
              angle={-45} 
              textAnchor="end" 
              interval={0} 
              height={80}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="no-streaming-original" fill="#8884d8" />
            <Bar dataKey="streaming-original" fill="#82ca9d" />
            <Bar dataKey="streaming-rag" fill="#ffc658" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default BenchmarkCharts;