'use client';

interface AgentState {
  id: string;
  status: 'idle' | 'thinking' | 'completed' | 'error';
  response: string;
  model: string;
}

interface AgentsSidebarProps {
  agents: AgentState[];
}

const statusColors = {
  idle: 'bg-gray-400',
  thinking: 'bg-blue-500 animate-pulse',
  completed: 'bg-green-500',
  error: 'bg-red-500',
};

const statusLabels = {
  idle: 'Idle',
  thinking: 'Thinking...',
  completed: 'Completed',
  error: 'Error',
};

export default function AgentsSidebar({ agents }: AgentsSidebarProps) {
  return (
    <div className="w-96 bg-gray-50 border-l border-gray-200 overflow-y-auto">
      <div className="p-4 bg-white border-b border-gray-200">
        <h2 className="text-lg font-bold text-gray-900">Worker Agents</h2>
        <p className="text-xs text-gray-500 mt-1">5 agents working in parallel</p>
      </div>

      <div className="p-4 space-y-3">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className="bg-white rounded-lg border border-gray-200 p-3"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${statusColors[agent.status]}`} />
                <span className="font-medium text-sm text-gray-900">
                  {agent.id}
                </span>
              </div>
              <span className="text-xs text-gray-500">
                {statusLabels[agent.status]}
              </span>
            </div>

            <p className="text-xs text-gray-500 mb-2">
              Model: {agent.model}
            </p>

            {agent.response && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                <p className="text-xs text-gray-700 line-clamp-4">
                  {agent.response}
                </p>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="p-4 bg-white border-t border-gray-200">
        <div className="text-xs text-gray-500">
          <p className="font-medium mb-1">Status Legend:</p>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              <span>Thinking</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span>Completed</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              <span>Error</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
