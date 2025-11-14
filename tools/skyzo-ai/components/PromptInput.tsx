'use client';

import { useState } from 'react';

interface PromptInputProps {
  onSubmit: (prompt: string) => void;
}

export default function PromptInput({ onSubmit }: PromptInputProps) {
  const [prompt, setPrompt] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) {
      onSubmit(prompt);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Skyzo-AI
          </h1>
          <p className="text-lg text-gray-600">
            Multi-Agent LLM System
          </p>
          <p className="text-sm text-gray-500 mt-2">
            5 agents working in parallel, coordinated by a leader
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
              Enter your prompt
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full h-40 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              placeholder="Example: Find gaps in the following theorem..."
            />
            <p className="text-sm text-gray-500 mt-2">
              {prompt.length} characters
            </p>
          </div>

          <button
            type="submit"
            disabled={!prompt.trim()}
            className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Start Analysis
          </button>
        </form>

        <div className="mt-8 p-4 bg-blue-50 rounded-lg">
          <h3 className="font-medium text-blue-900 mb-2">How it works:</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Your prompt is sent to a leader agent</li>
            <li>• The leader distributes it to 5 worker agents in parallel</li>
            <li>• Agents analyze from different perspectives</li>
            <li>• The leader coordinates and synthesizes their responses</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
