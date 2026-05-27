import React from 'react';

const modes = [
  { id: 'conservative', label: 'Conservative', desc: 'Tools only on explicit request (recommended for small models)' },
  { id: 'balanced', label: 'Balanced', desc: 'Smart auto + guardrails' },
  { id: 'aggressive', label: 'Aggressive', desc: 'Full tool freedom (power users)' },
];

interface Props {
  current: string;
  onChange: (mode: string) => void;
}

export const ToolModeSelector: React.FC<Props> = ({ current, onChange }) => {
  return (
    <div className="flex gap-2">
      {modes.map((mode) => (
        <button
          key={mode.id}
          onClick={() => onChange(mode.id)}
          className={`px-3 py-1 text-xs rounded-lg border transition-all
            ${current === mode.id 
              ? 'bg-white text-black border-white' 
              : 'border-white/20 hover:border-white/50 text-white/70'}`}
          title={mode.desc}
        >
          {mode.label}
        </button>
      ))}
    </div>
  );
};
