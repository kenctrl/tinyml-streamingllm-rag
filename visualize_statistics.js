import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';

const processData = (scores, avgGenTimes, avgTokensPerSecond) => {
  const benchmarks = [
    'coding', 'extraction', 'humanities', 
    'math', 'reasoning', 'roleplay', 
    'stem', 'writing'
  ];

  return benchmarks.map(benchmark => {
    const noStreamingKey = `${benchmark}_bench.jsonl-no-streaming-original.txt`;
    const streamingOriginalKey = `${benchmark}_bench.jsonl-streaming-original.txt`;
    const streamingRagKey = `${benchmark}_bench.jsonl-streaming-rag.txt`;

    return {
      benchmark,
      noStreamingScore: scores[noStreamingKey] ? 
        (scores[noStreamingKey].reduce((a, b) => a + b, 0) / scores[noStreamingKey].length).toFixed(2) : 0,
      streamingOriginalScore: scores[streamingOriginalKey] ? 
        (scores[streamingOriginalKey].reduce((a, b) => a + b, 0) / scores[streamingOriginalKey].length).toFixed(2) : 0,
      streamingRagScore: scores[streamingRagKey] ? 
        (scores[streamingRagKey].reduce((a, b) => a + b, 0) / scores[streamingRagKey].length).toFixed(2) : 0,

      noStreamingGenTime: avgGenTimes[noStreamingKey] ? avgGenTimes[noStreamingKey].toFixed(2) : 0,
      streamingOriginalGenTime: avgGenTimes[streamingOriginalKey] ? avgGenTimes[streamingOriginalKey].toFixed(2) : 0,
      streamingRagGenTime: avgGenTimes[streamingRagKey] ? avgGenTimes[streamingRagKey].toFixed(2) : 0,

      noStreamingTokensPerSec: avgTokensPerSecond[noStreamingKey] ? avgTokensPerSecond[noStreamingKey].toFixed(2) : 0,
      streamingOriginalTokensPerSec: avgTokensPerSecond[streamingOriginalKey] ? avgTokensPerSecond[streamingOriginalKey].toFixed(2) : 0,
      streamingRagTokensPerSec: avgTokensPerSecond[streamingRagKey] ? avgTokensPerSecond[streamingRagKey].toFixed(2) : 0,
    };
  });
};

const BenchmarkCharts = () => {
  const scores = {"coding_bench.jsonl-no-streaming-original.txt":[8.5,8.0,7.0,8.5,7.5,1.0,0.5,1.0,8.5,2.0,3.0,1.0,0.5,1.0,1.0,1.0,0.5,0.5,1.0,1.0],"coding_bench.jsonl-streaming-original.txt":[9.5,8.5,7.5,5.0,8.0,6.0,7.0,7.0,9.0,8.0,6.5,5.5,7.5,7.0,6.5,5.0,8.0,8.5,6.0,6.0],"coding_bench.jsonl-streaming-rag.txt":[9.5,8.0,9.0,8.5,7.0,6.5,8.0,7.5,6.0,8.0,6.5,5.5,7.0,5.5,7.5,8.0,7.0,7.5,6.0,9.0]};
  const avgGenTimes = {"coding_bench.jsonl-no-streaming-original.txt":2.95,"coding_bench.jsonl-streaming-original.txt":23.85,"coding_bench.jsonl-streaming-rag.txt":72.29};
  const avgTokensPerSecond = {"coding_bench.jsonl-no-streaming-original.txt":9.76,"coding_bench.jsonl-streaming-original.txt":0.95,"coding_bench.jsonl-streaming-rag.txt":1.32};

  const processedData = processData(scores, avgGenTimes, avgTokensPerSecond);

  return (
    <div className="space-y-6 p-4">
      {/* Scores Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Benchmark Scores by Category</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="benchmark" />
              <YAxis domain={[0, 10]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="noStreamingScore" fill="#8884d8" name="No Streaming" />
              <Bar dataKey="streamingOriginalScore" fill="#82ca9d" name="Streaming Original" />
              <Bar dataKey="streamingRagScore" fill="#ffc658" name="Streaming RAG" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Average Generation Time Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Average Generation Time (seconds)</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="benchmark" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="noStreamingGenTime" fill="#8884d8" name="No Streaming" />
              <Bar dataKey="streamingOriginalGenTime" fill="#82ca9d" name="Streaming Original" />
              <Bar dataKey="streamingRagGenTime" fill="#ffc658" name="Streaming RAG" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Average Tokens per Second Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Average Tokens per Second</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={processedData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="benchmark" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="noStreamingTokensPerSec" fill="#8884d8" name="No Streaming" />
              <Bar dataKey="streamingOriginalTokensPerSec" fill="#82ca9d" name="Streaming Original" />
              <Bar dataKey="streamingRagTokensPerSec" fill="#ffc658" name="Streaming RAG" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
};

export default BenchmarkCharts;