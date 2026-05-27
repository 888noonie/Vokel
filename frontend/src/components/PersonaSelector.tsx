import React from 'react';

const personas = [
  { id: 'cosmic', label: 'Cosmic Explorer', emoji: '🌌', prompt: 'You are a curious cosmic explorer...' },
  { id: 'analyst', label: 'Sharp Analyst', emoji: '🔬', prompt: 'You are a precise, evidence-based analyst...' },
  { id: 'companion', label: 'Warm Companion', emoji: '🫂', prompt: 'You are a warm, empathetic companion...' },
  { id: 'inventor', label: 'Playful Inventor', emoji: '🧪', prompt: 'You are a playful, inventive mind...' },
  { id: 'philosopher', label: 'Deep Philosopher', emoji: '🧠', prompt: 'You are a thoughtful philosopher...' },
];

interface Props {
  current: string;
  onChange: (id: string, prompt: string) => void;
}

export const PersonaSelector: React.FC<Props> = ({ current, onChange }) => {
  return (
    <div className="flex gap-2 flex-wrap">
      {personas.map((p) => (
        <button
          key={p.id}
          onClick={() => onChange(p.id, p.prompt)}
          className={`px-4 py-1.5 rounded-full text-sm flex items-center gap-2 transition-all
            ${current === p.id 
              ? 'bg-white text-black shadow-lg scale-105' 
              : 'bg-white/10 hover:bg-white/20 text-white'}`}
        >
          <span>{p.emoji}</span>
          <span>{p.label}</span>
        </button>
      ))}
    </div>
  );
};
