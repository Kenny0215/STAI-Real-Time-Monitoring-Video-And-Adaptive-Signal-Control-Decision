import React from 'react';

const FLASK_URL = 'http://127.0.0.1:5000';

interface StreamImgProps {
  laneKey:    string;
  className?: string;
  isActive:   boolean;
  streamKey:  number;
}

export const StreamImg = ({ laneKey, className = '', isActive, streamKey }: StreamImgProps) => {
  if (!isActive) {
    return (
      <div className={`flex items-center justify-center bg-black ${className}`}>
        <p className="text-slate-600 text-xs">Stream stopped</p>
      </div>
    );
  }

  return (
    <img
      key={`${laneKey}-${streamKey}`}
      src={`${FLASK_URL}/api/stream-local/${laneKey}`}
      alt={`${laneKey} stream`}
      className={className}
      draggable={false}
      onLoad={() => console.log(`[Stream] ${laneKey} loaded successfully`)}
      onError={(e) => {
        console.error(`[Stream] ${laneKey} FAILED to load`, e);
      }}
    />
  );
};