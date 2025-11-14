'use client';

import { useState, useEffect } from 'react';
import AgentsSidebar from './AgentsSidebar';

interface ConversationViewProps {
  initialPrompt: string;
}

interface AgentState {
  id: string;
  status: 'idle' | 'thinking' | 'completed' | 'error';
  response: string;
  model: string;
}

export default function ConversationView({ initialPrompt }: ConversationViewProps) {
  const [leaderMessage, setLeaderMessage] = useState('');
  const [agents, setAgents] = useState<AgentState[]>([
    { id: 'Agent-1', status: 'idle', response: '', model: 'gpt-4-turbo' },
    { id: 'Agent-2', status: 'idle', response: '', model: 'claude-3-opus' },
    { id: 'Agent-3', status: 'idle', response: '', model: 'gemini-pro' },
    { id: 'Agent-4', status: 'idle', response: '', model: 'gpt-4-turbo' },
    { id: 'Agent-5', status: 'idle', response: '', model: 'claude-3-sonnet' },
  ]);
  const [isProcessing, setIsProcessing] = useState(true);

  useEffect(() => {
    // Start the query processing
    processQuery();
  }, []);

  const processQuery = async () => {
    try {
      // Set all agents to thinking
      setAgents(prev => prev.map(a => ({ ...a, status: 'thinking' as const })));
      setLeaderMessage('Leader: Distributing query to all 5 agents...\n\n');

      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: initialPrompt }),
      });

      if (!response.ok) {
        throw new Error('Failed to process query');
      }

      const data = await response.json();
      
      // Update agents with their responses
      if (data.agentResponses) {
        setAgents(prev => prev.map((agent, idx) => ({
          ...agent,
          status: 'completed' as const,
          response: data.agentResponses[idx] || 'No response',
        })));
      }

      // Update leader message
      if (data.leaderResponse) {
        setLeaderMessage(prev => prev + '\n\nLeader: ' + data.leaderResponse);
      }

      setIsProcessing(false);
    } catch (error) {
      console.error('Error processing query:', error);
      setLeaderMessage('Error: Failed to process query. Make sure OPENROUTER_API_KEY is set.');
      setAgents(prev => prev.map(a => ({ ...a, status: 'error' as const })));
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen">
      {/* Main conversation area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="bg-white border-b border-gray-200 p-4">
          <h1 className="text-xl font-bold text-gray-900">Skyzo-AI Conversation</h1>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {/* User message */}
          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-sm font-medium text-blue-900 mb-2">You:</p>
            <p className="text-gray-800">{initialPrompt}</p>
          </div>

          {/* Leader message */}
          <div className="bg-white rounded-lg p-4 border border-gray-200">
            <p className="text-sm font-medium text-gray-900 mb-2">Leader Agent:</p>
            <div className="text-gray-800 whitespace-pre-wrap">
              {leaderMessage || 'Processing...'}
              {isProcessing && (
                <span className="inline-block ml-2 animate-pulse">●</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Agents sidebar */}
      <AgentsSidebar agents={agents} />
    </div>
  );
}
