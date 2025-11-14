# Skyzo-AI Quick Start Guide

## What You Have Now

A **fully functional minimum MVP** of a multi-agent LLM system where:
- 5 worker agents process queries in parallel
- 1 leader agent coordinates and synthesizes responses
- Clean web UI with real-time status updates
- Works with or without OpenRouter API key

## Running the Application

### 1. Install Dependencies
```bash
cd tools/skyzo-ai
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

### 3. Open Browser
Navigate to: http://localhost:3000

## Using the Application

### Basic Flow
1. **Enter a prompt** in the text area (e.g., "Find gaps in this theorem...")
2. **Click "Start Analysis"**
3. **Watch the agents work:**
   - Left side: Leader's synthesized response
   - Right sidebar: All 5 agents processing in parallel
4. **See results** as agents complete their analysis

### Mock Mode (Default)
The app works immediately without any API key:
- Uses simulated agent responses
- Demonstrates the full UI and flow
- Perfect for testing and understanding the system

### Real Mode (Optional)
To use actual LLM models:

1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Create `.env.local` file:
   ```bash
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```
3. Restart the server
4. Now agents use real models:
   - Agent 1: GPT-4 Turbo
   - Agent 2: Claude 3 Opus
   - Agent 3: Gemini Pro
   - Agent 4: GPT-4 Turbo
   - Agent 5: Claude 3 Sonnet

## Example Prompts to Try

1. **Mathematical Analysis:**
   ```
   Find gaps in the following theorem: All prime numbers greater than 2 are odd
   ```

2. **Technical Question:**
   ```
   What are the key challenges in implementing quantum error correction?
   ```

3. **Problem Solving:**
   ```
   Analyze the trade-offs between different database architectures for a social media application
   ```

## Architecture Overview

```
┌─────────────┐
│    User     │
│   Prompt    │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Leader Agent   │ ← Maintains full context
│  (Claude Opus)  │ ← Coordinates all agents
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┬────────┐
    ▼         ▼        ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│Agent 1│ │Agent 2│ │Agent 3│ │Agent 4│ │Agent 5│
│ GPT-4 │ │Claude │ │Gemini │ │ GPT-4 │ │Claude │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘
    │         │        │        │        │
    └─────────┴────────┴────────┴────────┘
                     │
                     ▼
           ┌──────────────────┐
           │ Leader Synthesis │
           │   & Response     │
           └──────────────────┘
```

## Key Features

✅ **Simple Interface** - Just enter text and go  
✅ **Parallel Processing** - All 5 agents work simultaneously  
✅ **Real-time Updates** - See agents thinking/completing  
✅ **Leader Coordination** - Synthesizes multiple perspectives  
✅ **Mock Mode** - Test without API costs  
✅ **Production Ready** - Can deploy immediately  

## File Structure

```
tools/skyzo-ai/
├── app/
│   ├── page.tsx              # Main page (handles state)
│   ├── layout.tsx            # Root layout
│   └── api/
│       └── query/
│           └── route.ts      # API endpoint (agents logic)
├── components/
│   ├── PromptInput.tsx       # Landing page with text input
│   ├── ConversationView.tsx  # Main conversation display
│   └── AgentsSidebar.tsx     # Sidebar with agent status
├── README.md                 # Full documentation
├── CONCEPT.md                # System design details
├── IMPLEMENTATION.md         # Technical implementation guide
└── TODO.md                   # Future roadmap
```

## API Endpoint

**POST** `/api/query`

```json
Request:
{
  "prompt": "Your question here"
}

Response:
{
  "leaderResponse": "Synthesized answer from all agents...",
  "agentResponses": [
    "Agent 1 response...",
    "Agent 2 response...",
    "Agent 3 response...",
    "Agent 4 response...",
    "Agent 5 response..."
  ]
}
```

## Building for Production

```bash
npm run build
npm start
```

Deploy to Vercel, Netlify, or any Node.js hosting platform.

## What's Next?

See `TODO.md` for the full roadmap including:
- Real-time streaming with Server-Sent Events
- Agent-to-agent communication display
- Conversation history
- Advanced coordination strategies
- Custom agent configurations

## Troubleshooting

**Port already in use:**
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill
```

**Build fails:**
```bash
# Clear cache and rebuild
rm -rf .next node_modules package-lock.json
npm install
npm run build
```

**Can't see agent responses:**
- Check browser console for errors
- Ensure API endpoint is accessible at `/api/query`
- Verify agents are updating (check sidebar status)

## Success!

You now have a working multi-agent LLM system. Test it, experiment with different prompts, and see how multiple AI perspectives can provide more comprehensive answers than a single model.

For questions or issues, refer to:
- README.md - Complete documentation
- CONCEPT.md - System architecture
- IMPLEMENTATION.md - Technical details
