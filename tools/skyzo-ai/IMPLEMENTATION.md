# Skyzo-AI Implementation Guide

## Technology Stack

### Frontend
- **Framework**: Next.js 14+ (App Router)
- **Language**: TypeScript
- **UI Components**: React 18+
- **Styling**: Tailwind CSS
- **Real-time Updates**: Server-Sent Events (SSE) or WebSockets
- **State Management**: React Context + Hooks / Zustand
- **Markdown Rendering**: react-markdown

### Backend
- **Runtime**: Node.js 18+
- **API Routes**: Next.js API Routes
- **LLM Provider**: OpenRouter API
- **Streaming**: OpenAI-compatible streaming API
- **Error Handling**: Custom middleware

### Development Tools
- **Package Manager**: npm/pnpm/yarn
- **Linting**: ESLint
- **Formatting**: Prettier
- **Type Checking**: TypeScript strict mode

## Application Architecture

### Directory Structure

```
skyzo-ai/
├── app/
│   ├── layout.tsx                 # Root layout
│   ├── page.tsx                   # Landing/prompt page
│   ├── conversation/
│   │   └── [id]/
│   │       └── page.tsx           # Conversation thread page
│   └── api/
│       ├── query/
│       │   └── route.ts           # Main query endpoint
│       ├── stream/
│       │   └── route.ts           # SSE endpoint for real-time updates
│       └── conversation/
│           └── [id]/
│               └── route.ts       # Get conversation history
├── components/
│   ├── PromptInput.tsx            # Initial prompt interface
│   ├── ConversationThread.tsx    # Main conversation display
│   ├── LeaderMessages.tsx         # Leader responses
│   ├── AgentsSidebar.tsx          # Sidebar with agent activity
│   ├── AgentCard.tsx              # Individual agent display
│   ├── StreamingMessage.tsx       # Streaming text component
│   └── ChainOfThought.tsx         # CoT visualization
├── lib/
│   ├── openrouter.ts              # OpenRouter API client
│   ├── agents/
│   │   ├── leader.ts              # Leader agent logic
│   │   ├── worker.ts              # Worker agent logic
│   │   └── coordinator.ts         # Coordination logic
│   ├── types.ts                   # TypeScript types
│   └── utils.ts                   # Utility functions
├── store/
│   └── conversation.ts            # Conversation state management
└── public/
    └── assets/                    # Static assets
```

## Core Components

### 1. Landing Page (Prompt Input)

**File**: `app/page.tsx`

```typescript
// Initial interface where user submits their query
- Large text area for prompt input
- Submit button
- Previous conversations list (optional)
- Clear instructions on system capabilities
```

**Features**:
- Auto-save draft prompts
- Prompt templates for common use cases
- Character counter
- Examples/suggestions

### 2. Conversation Page

**File**: `app/conversation/[id]/page.tsx`

**Layout**:
```
┌────────────────────────────────────────┬──────────────────┐
│                                        │   Agent Panel    │
│  Main Conversation Thread              │                  │
│  (Leader Messages)                     │   Agent A: ●     │
│                                        │   [streaming...] │
│  ┌─────────────────────────────────┐  │                  │
│  │ User: Find gaps in theorem      │  │   Agent B: ●     │
│  └─────────────────────────────────┘  │   [thinking...]  │
│                                        │                  │
│  ┌─────────────────────────────────┐  │   Agent C: ✓     │
│  │ Leader: Analyzing with 5 agents │  │   [completed]    │
│  │ ...streaming response...        │  │                  │
│  └─────────────────────────────────┘  │   Agent D: ●     │
│                                        │   [streaming...] │
│  ┌─────────────────────────────────┐  │                  │
│  │ [Input box for follow-up]       │  │   Agent E: ●     │
│  └─────────────────────────────────┘  │   [streaming...] │
└────────────────────────────────────────┴──────────────────┘
```

### 3. Agents Sidebar

**File**: `components/AgentsSidebar.tsx`

**Displays**:
- Agent status (thinking, streaming, completed, idle)
- Real-time streaming of agent responses
- Chain-of-thought reasoning (expandable)
- Agent-to-agent messages
- Leader instructions to agents
- Visual indicators for agent state

**Features**:
- Collapsible/expandable agent cards
- Filter by agent status
- Search through agent responses
- Export agent logs

### 4. Conversation Thread

**File**: `components/ConversationThread.tsx`

**Displays**:
- User messages
- Leader responses (main conversation)
- Timestamps
- Loading states
- Error states

**Features**:
- Markdown rendering
- Code syntax highlighting
- Copy to clipboard
- Message reactions (optional)

## API Implementation

### OpenRouter Integration

**File**: `lib/openrouter.ts`

```typescript
interface OpenRouterConfig {
  apiKey: string;
  baseURL: string;
  defaultModel: string;
}

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface StreamOptions {
  model: string;
  messages: Message[];
  stream: boolean;
  temperature?: number;
}

class OpenRouterClient {
  // Stream response from OpenRouter
  async streamCompletion(options: StreamOptions): Promise<ReadableStream>
  
  // Non-streaming completion
  async completion(options: StreamOptions): Promise<string>
}
```

**Models Configuration**:
- Leader: GPT-4, Claude 3 Opus, or similar high-capability model
- Workers: Mix of top-tier models (GPT-4, Claude 3 Opus/Sonnet, Gemini Pro, etc.)

### Query Processing Flow

**File**: `app/api/query/route.ts`

```typescript
POST /api/query
Body: { prompt: string, conversationId?: string }

Flow:
1. Validate and sanitize input
2. Initialize/load conversation context
3. Create leader context with full history
4. Distribute query to all 5 worker agents (parallel)
5. Stream responses to client
6. Leader processes agent responses
7. Leader generates coordination messages
8. Leader formulates final response
9. Return conversation ID and initial responses
```

### Real-time Updates

**File**: `app/api/stream/route.ts`

```typescript
GET /api/stream?conversationId=xxx

Implements Server-Sent Events (SSE):
- Event: agent-update
  Data: { agentId, status, message, timestamp }
  
- Event: leader-message
  Data: { content, timestamp, agentReferences }
  
- Event: coordination
  Data: { fromAgent, toAgent, message }
  
- Event: completion
  Data: { conversationId, summary }
  
- Event: error
  Data: { error, agentId }
```

## Agent Logic Implementation

### Worker Agent

**File**: `lib/agents/worker.ts`

```typescript
interface WorkerAgent {
  id: string;
  model: string;
  systemPrompt: string;
  status: 'idle' | 'thinking' | 'streaming' | 'completed' | 'error';
}

class WorkerAgentRunner {
  async processQuery(
    agent: WorkerAgent,
    query: string,
    context: string[]
  ): Promise<AsyncIterator<string>>
  
  async receiveCoordination(
    message: string,
    fromLeader: boolean
  ): Promise<void>
}
```

**System Prompt Template**:
```
You are Agent {ID} in a multi-agent collaborative system. You are working 
alongside 4 other agents on the same problem, coordinated by a leader agent.

Your role:
- Provide your unique perspective and analysis
- Think independently but be aware others are working in parallel
- Your responses may be shared with other agents
- You may receive updates about other agents' progress

Current task: {TASK}
```

### Leader Agent

**File**: `lib/agents/leader.ts`

```typescript
interface LeaderAgent {
  fullContext: Message[];
  agentResponses: Map<string, string[]>;
  coordinationHistory: CoordinationMessage[];
}

class LeaderAgentRunner {
  // Analyze all agent responses
  async analyzeResponses(
    responses: Map<string, string>
  ): Promise<Analysis>
  
  // Generate coordination messages for agents
  async generateCoordination(
    analysis: Analysis
  ): Promise<CoordinationMessage[]>
  
  // Synthesize final response
  async synthesizeResponse(
    analysis: Analysis
  ): Promise<string>
  
  // Determine if more information is needed
  shouldContinue(analysis: Analysis): boolean
}
```

### Coordinator

**File**: `lib/agents/coordinator.ts`

```typescript
class AgentCoordinator {
  // Distribute query to all agents
  async distributeQuery(
    query: string,
    agents: WorkerAgent[]
  ): Promise<void>
  
  // Collect responses with timeout
  async collectResponses(
    timeout: number
  ): Promise<Map<string, string>>
  
  // Send coordination messages
  async sendCoordination(
    messages: CoordinationMessage[]
  ): Promise<void>
  
  // Monitor agent status
  getAgentStatuses(): Map<string, AgentStatus>
}
```

## State Management

### Conversation Store

**File**: `store/conversation.ts`

```typescript
interface ConversationState {
  id: string;
  messages: Message[];
  agentStates: Map<string, AgentState>;
  leaderState: LeaderState;
  isProcessing: boolean;
}

interface AgentState {
  id: string;
  status: AgentStatus;
  messages: string[];
  chainOfThought: string[];
  coordinationReceived: string[];
}

// Zustand store or Context
const useConversation = create<ConversationState>((set) => ({
  // State and actions
}));
```

## Real-time UI Updates

### Streaming Implementation

```typescript
// Client-side streaming consumer
const useAgentStream = (conversationId: string) => {
  useEffect(() => {
    const eventSource = new EventSource(`/api/stream?id=${conversationId}`);
    
    eventSource.addEventListener('agent-update', (e) => {
      const data = JSON.parse(e.data);
      updateAgentState(data.agentId, data);
    });
    
    eventSource.addEventListener('leader-message', (e) => {
      const data = JSON.parse(e.data);
      appendLeaderMessage(data);
    });
    
    eventSource.addEventListener('coordination', (e) => {
      const data = JSON.parse(e.data);
      showCoordination(data);
    });
    
    return () => eventSource.close();
  }, [conversationId]);
};
```

## UI/UX Features

### Agent Visualization

1. **Status Indicators**:
   - 🔵 Thinking (pulsing blue)
   - 🟢 Streaming (animated green)
   - ✅ Completed (checkmark)
   - ⭕ Idle (gray)
   - ❌ Error (red)

2. **Progress Animation**:
   - Typing indicators for streaming
   - Progress bars for long operations
   - Smooth transitions between states

3. **Expandable Details**:
   - Click agent to expand full response
   - View chain-of-thought reasoning
   - See coordination messages

### Leader Messages

1. **Response Formatting**:
   - Markdown support
   - Syntax highlighting for code
   - Math equation rendering (optional)
   - Reference tags to agent contributions

2. **Interaction**:
   - Inline agent references (clickable)
   - Follow-up input always available
   - Copy/share responses

## Data Persistence

### Storage Options

1. **Browser LocalStorage** (MVP):
   - Store conversations locally
   - Quick to implement
   - No backend needed initially

2. **Database** (Production):
   - PostgreSQL / MongoDB
   - Store conversations, agent logs
   - User accounts (optional)

### Schema

```typescript
interface Conversation {
  id: string;
  userId?: string;
  createdAt: Date;
  updatedAt: Date;
  messages: Message[];
  agentLogs: AgentLog[];
  status: 'active' | 'completed';
}

interface AgentLog {
  agentId: string;
  timestamp: Date;
  type: 'response' | 'coordination' | 'status';
  content: string;
}
```

## Environment Configuration

```bash
# .env.local
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Agent Models
LEADER_MODEL=anthropic/claude-3-opus
AGENT_1_MODEL=openai/gpt-4-turbo
AGENT_2_MODEL=anthropic/claude-3-opus
AGENT_3_MODEL=google/gemini-pro
AGENT_4_MODEL=openai/gpt-4-turbo
AGENT_5_MODEL=anthropic/claude-3-sonnet

# Configuration
MAX_TOKENS=4000
TEMPERATURE=0.7
TIMEOUT=60000
```

## Error Handling

```typescript
// Handle agent failures gracefully
class AgentError extends Error {
  constructor(
    public agentId: string,
    public originalError: Error
  ) {
    super(`Agent ${agentId} error: ${originalError.message}`);
  }
}

// Retry logic for failed agents
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3
): Promise<T> {
  // Implementation
}

// Partial success handling
// Leader can still synthesize response even if some agents fail
```

## Performance Optimization

1. **Parallel Agent Execution**:
   - Use `Promise.all()` for concurrent agent queries
   - Set reasonable timeouts

2. **Streaming**:
   - Stream responses as they arrive
   - Don't wait for all agents to complete

3. **Caching**:
   - Cache conversation context
   - Reuse agent connections

4. **Rate Limiting**:
   - Implement rate limiting for API calls
   - Handle API quota gracefully

## Security Considerations

1. **API Key Protection**:
   - Never expose keys client-side
   - Use environment variables
   - Implement server-side proxy

2. **Input Validation**:
   - Sanitize user input
   - Limit message length
   - Prevent injection attacks

3. **Rate Limiting**:
   - Limit requests per user
   - Prevent abuse

4. **Content Filtering**:
   - Implement content moderation
   - Handle inappropriate content

## Testing Strategy

1. **Unit Tests**:
   - Agent logic
   - Coordinator logic
   - API routes

2. **Integration Tests**:
   - Full query flow
   - Streaming functionality
   - Error handling

3. **E2E Tests**:
   - User workflows
   - UI interactions
   - Real-time updates

4. **Mock OpenRouter**:
   - Create mock responses for testing
   - Test without API calls

## Deployment

### Vercel (Recommended)
- Native Next.js support
- Edge functions for API routes
- Automatic scaling
- Environment variable management

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
CMD ["npm", "start"]
```

### Environment Setup
- Configure OpenRouter API key
- Set model preferences
- Configure timeouts and limits
