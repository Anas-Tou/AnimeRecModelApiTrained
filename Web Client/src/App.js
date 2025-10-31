import React, { useState } from 'react';
import SimilarAnimePage from './SimilarAnimePage';
import GenreRecommendationPage from './GenreRecommendationPage';
import './App.css';

function App() {
  const [page, setPage] = useState('similar');
  const apiUrl = 'http://localhost:8000';

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="site-header">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <h1 className="text-2xl font-bold text-gray-900">Anime Recommendations</h1>
            <nav className="flex space-x-2">
              <button
                onClick={() => setPage('similar')}
                className={`nav-button ${page === 'similar' ? 'active' : ''}`}
              >
                Similar Anime
              </button>
              <button
                onClick={() => setPage('genre')}
                className={`nav-button ${page === 'genre' ? 'active' : ''}`}
              >
                Genre Recommendations
              </button>
            </nav>
          </div>
        </div>
      </header>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="bg-white rounded-lg shadow-sm p-6">
          {page === 'similar' ? <SimilarAnimePage apiUrl={apiUrl} /> : <GenreRecommendationPage apiUrl={apiUrl} />}
        </div>
      </main>
      <footer className="mt-8 py-4 text-center text-gray-600 text-sm">
        <p>Made with ❤️ for anime fans</p>
      </footer>
    </div>
  );
}

export default App;