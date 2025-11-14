import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { prompt } = await request.json();

    if (!prompt) {
      return NextResponse.json(
        { error: 'Prompt is required' },
        { status: 400 }
      );
    }

    // Check if OpenRouter API key is available
    const apiKey = process.env.OPENROUTER_API_KEY;
    
    if (!apiKey) {
      // Return mock responses for MVP testing
      return NextResponse.json({
        leaderResponse: getMockLeaderResponse(prompt),
        agentResponses: getMockAgentResponses(prompt),
      });
    }

    // If API key exists, use real OpenRouter API
    const agentResponses = await Promise.all([
      queryAgent('Agent-1', prompt, 'openai/gpt-4-turbo', apiKey),
      queryAgent('Agent-2', prompt, 'anthropic/claude-3-opus', apiKey),
      queryAgent('Agent-3', prompt, 'google/gemini-pro', apiKey),
      queryAgent('Agent-4', prompt, 'openai/gpt-4-turbo', apiKey),
      queryAgent('Agent-5', prompt, 'anthropic/claude-3-sonnet', apiKey),
    ]);

    // Leader synthesizes responses
    const leaderResponse = await synthesizeResponses(prompt, agentResponses, apiKey);

    return NextResponse.json({
      leaderResponse,
      agentResponses,
    });
  } catch (error) {
    console.error('Error processing query:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

async function queryAgent(
  agentId: string,
  prompt: string,
  model: string,
  apiKey: string
): Promise<string> {
  const systemPrompt = `You are ${agentId} in a multi-agent collaborative system. You are working alongside 4 other agents on the same problem, coordinated by a leader agent.

Your role:
- Provide your unique perspective and analysis
- Think independently but be aware others are working in parallel
- Your responses may be shared with other agents

Be concise but thorough in your analysis.`;

  try {
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/danielfebrero/mathematics',
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: prompt },
        ],
        max_tokens: 500,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenRouter API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || 'No response';
  } catch (error) {
    console.error(`Error querying ${agentId}:`, error);
    return `Error: Failed to get response from ${agentId}`;
  }
}

async function synthesizeResponses(
  prompt: string,
  agentResponses: string[],
  apiKey: string
): Promise<string> {
  const leaderPrompt = `You are the Leader Agent coordinating 5 worker agents. They have all analyzed the following prompt:

"${prompt}"

Here are their responses:

Agent-1: ${agentResponses[0]}

Agent-2: ${agentResponses[1]}

Agent-3: ${agentResponses[2]}

Agent-4: ${agentResponses[3]}

Agent-5: ${agentResponses[4]}

Your task:
1. Synthesize their perspectives into a coherent response
2. Highlight areas of agreement and disagreement
3. Provide a comprehensive answer to the user's prompt
4. Note any particularly insightful observations from individual agents`;

  try {
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/danielfebrero/mathematics',
      },
      body: JSON.stringify({
        model: 'anthropic/claude-3-opus',
        messages: [
          { role: 'user', content: leaderPrompt },
        ],
        max_tokens: 1000,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenRouter API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.choices[0]?.message?.content || 'Unable to synthesize responses';
  } catch (error) {
    console.error('Error synthesizing responses:', error);
    return 'Error: Failed to synthesize agent responses';
  }
}

// Mock responses for MVP testing without API key
function getMockAgentResponses(prompt: string): string[] {
  return [
    `Agent-1 (GPT-4): Analyzing the prompt "${prompt.substring(0, 50)}..." - I notice several key aspects that need consideration. The main challenge appears to be in the structural approach. I recommend breaking this down into smaller components.`,
    
    `Agent-2 (Claude-3): From my perspective, this problem requires careful examination of the underlying assumptions. I've identified three potential approaches, with the second showing the most promise based on logical consistency.`,
    
    `Agent-3 (Gemini): I have a different take on this. While the previous agents focused on structure, I believe the key lies in the methodological framework. There might be a gap in the reasoning that needs to be addressed first.`,
    
    `Agent-4 (GPT-4): Building on the analysis, I've found some interesting patterns. However, I disagree with some of the initial assumptions. My investigation reveals that we should consider alternative interpretations.`,
    
    `Agent-5 (Claude-3): After reviewing the problem comprehensively, I can confirm that multiple valid approaches exist. The optimal solution likely combines elements from different perspectives. I recommend proceeding with caution on certain aspects.`,
  ];
}

function getMockLeaderResponse(prompt: string): string {
  return `Analysis of "${prompt.substring(0, 80)}..."

AGENT COORDINATION SUMMARY:
After distributing your query to all 5 worker agents, here's what we discovered:

CONSENSUS AREAS:
• All agents agree this requires careful, structured analysis
• Multiple approaches were identified and validated
• There are underlying assumptions that need examination

DIVERGENT PERSPECTIVES:
• Agent-1 and Agent-4 focused on structural decomposition
• Agent-2 emphasized logical consistency
• Agent-3 highlighted methodological concerns
• Agent-5 took a comprehensive, balanced view

KEY INSIGHTS:
1. Agent-3's observation about methodological gaps is particularly important
2. Agent-4's challenge to initial assumptions opens new avenues
3. Agent-2 and Agent-5 converged on similar conclusions about multiple valid approaches

SYNTHESIZED RESPONSE:
Based on the collective intelligence of all agents, the most effective approach is to:
1. First address the methodological framework (per Agent-3)
2. Break down the problem into components (per Agent-1)
3. Apply logical consistency checks (per Agent-2)
4. Consider alternative interpretations (per Agent-4)
5. Combine elements from different perspectives (per Agent-5)

This multi-agent analysis reveals that the problem is more nuanced than it initially appears, and the solution benefits significantly from multiple perspectives working in parallel.

NOTE: This is a MOCK response. Set OPENROUTER_API_KEY environment variable for real multi-agent processing.`;
}
