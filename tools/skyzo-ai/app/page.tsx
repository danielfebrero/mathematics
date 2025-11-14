'use client';

import { useState } from 'react';
import PromptInput from '@/components/PromptInput';
import ConversationView from '@/components/ConversationView';

export default function Home() {
  const [conversationStarted, setConversationStarted] = useState(false);
  const [prompt, setPrompt] = useState('');

  const handleStartConversation = (userPrompt: string) => {
    setPrompt(userPrompt);
    setConversationStarted(true);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {!conversationStarted ? (
        <PromptInput onSubmit={handleStartConversation} />
      ) : (
        <ConversationView initialPrompt={prompt} />
      )}
    </div>
  );
}
