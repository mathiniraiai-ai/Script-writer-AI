import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import ScriptBrain from './pages/script-brain/ScriptBrain';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ScriptBrain />} />
        <Route path="/team/script-brain" element={<ScriptBrain />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
