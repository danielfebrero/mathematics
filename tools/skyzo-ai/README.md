# Skyzo-AI - Multi-Agent LLM System

A minimum viable product (MVP) demonstrating multi-agent LLM collaboration where 5 worker agents process queries in parallel, coordinated by a leader agent.

## Quick Start

### 1. Install Dependencies

```bash
cd tools/skyzo-ai
npm install
```

### 2. Run the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Features

- **Simple Text Input**: Enter any prompt or query
- **5 Worker Agents**: Running in parallel with different perspectives
- **Leader Agent**: Coordinates and synthesizes agent responses
- **Real-time UI**: See agent status and responses in sidebar
- **Mock Mode**: Works without API key for testing

## How It Works

1. **User Input**: Enter a prompt in the text area
2. **Distribution**: Leader sends the query to all 5 worker agents in parallel
3. **Agent Processing**: Each agent analyzes from their unique perspective
4. **Coordination**: Leader synthesizes all agent responses
5. **Response**: View coordinated results in the main thread and agent details in sidebar

## Using with OpenRouter API (Optional)

To use real LLM models instead of mock responses:

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Create a `.env.local` file:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

3. Restart the development server

### Configured Models

- **Agent 1**: OpenAI GPT-4 Turbo
- **Agent 2**: Anthropic Claude 3 Opus
- **Agent 3**: Google Gemini Pro
- **Agent 4**: OpenAI GPT-4 Turbo
- **Agent 5**: Anthropic Claude 3 Sonnet
- **Leader**: Anthropic Claude 3 Opus

## Project Structure

```
tools/skyzo-ai/
├── app/
│   ├── page.tsx                 # Main page with conversation state
│   └── api/
│       └── query/
│           └── route.ts         # API endpoint for query processing
├── components/
│   ├── PromptInput.tsx          # Initial prompt input screen
│   ├── ConversationView.tsx     # Main conversation display
│   └── AgentsSidebar.tsx        # Sidebar showing agent activity
├── CONCEPT.md                   # Detailed system concept
├── IMPLEMENTATION.md            # Full implementation guide
└── TODO.md                      # Development roadmap
```

## Architecture

```
User Query
    ↓
Leader Agent (full context)
    ↓
┌───────────────────────────────────┐
│   Distribute to 5 Worker Agents  │
│        (parallel execution)        │
└───────────────────────────────────┘
    ↓           ↓           ↓
 Agent-1    Agent-2    Agent-3  ...
    ↓           ↓           ↓
Agents respond with perspectives
    ↓
Leader coordinates:
  "Agent B suggests different approach"
  "Agent C has solution, challenge it"
    ↓
Leader synthesizes final response
```

## Development

### Build for Production

```bash
npm run build
npm start
```

### Linting

```bash
npm run lint
```

## Testing the MVP

1. Start the app with `npm run dev`
2. Enter a test prompt like:
   - "Find gaps in the following theorem: All prime numbers greater than 2 are odd"
   - "What are the key challenges in quantum computing?"
   - "Analyze the strengths and weaknesses of neural networks"
3. Watch as agents process in parallel
4. View synthesized response from the leader

## Mock Mode

Without an OpenRouter API key, the system runs in mock mode:
- Agents return simulated responses
- Leader provides a template synthesis
- All UI features work normally
- Perfect for testing the interface and flow

## Next Steps

See `TODO.md` for the full development roadmap including:
- Real-time streaming with SSE
- Enhanced agent visualization
- Agent-to-agent communication display
- Conversation history
- Advanced coordination strategies

## Documentation

- **CONCEPT.md**: Detailed explanation of the multi-agent architecture
- **IMPLEMENTATION.md**: Complete technical implementation guide
- **TODO.md**: Development roadmap with 9 phases

## License

Part of the [danielfebrero/mathematics](https://github.com/danielfebrero/mathematics) repository.
