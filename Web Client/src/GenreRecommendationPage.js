import React, { useState, useEffect } from 'react';

const CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours in milliseconds
const BATCH_SIZE = 3; // Number of parallel requests
const RATE_LIMIT_DELAY = 250; // 250ms between requests

const LoadingSpinner = () => (
  <div className="flex flex-col items-center justify-center py-12">
    <div className="relative">
      <div className="w-12 h-12 border-4 border-blue-200 rounded-full"></div>
      <div className="w-12 h-12 border-4 border-blue-600 rounded-full border-t-transparent animate-spin absolute top-0 left-0"></div>
    </div>
    <div className="mt-4 text-center">
      <div className="text-lg font-medium text-gray-900">Finding anime recommendations...</div>
      <div className="text-sm text-gray-500 mt-1">This may take a few moments</div>
    </div>
  </div>
);

const LoadingCard = () => (
  <div className="bg-white rounded-xl shadow-lg overflow-hidden transform transition-all duration-300 hover:shadow-xl hover:-translate-y-1">
    <div className="aspect-w-2 bg-gray-200 animate-pulse"></div>
    <div className="p-4 space-y-3">
      <div className="h-4 bg-gray-200 rounded animate-pulse w-3/4"></div>
      <div className="flex flex-wrap gap-1">
        <div className="h-6 bg-gray-200 rounded animate-pulse w-16"></div>
        <div className="h-6 bg-gray-200 rounded animate-pulse w-20"></div>
      </div>
      <div className="h-4 bg-gray-200 rounded animate-pulse w-1/4"></div>
    </div>
  </div>
);

const AnimeCard = ({ rec, imageUrl }) => {
  const handleClick = () => {
    // Create Google search URL with the anime name
    const searchQuery = encodeURIComponent(`${rec.name} anime`);
    const googleUrl = `https://www.google.com/search?q=${searchQuery}`;
    // Open in new tab
    window.open(googleUrl, '_blank', 'noopener,noreferrer');
  };

  return (
    <div 
      className="bg-white rounded-xl shadow-lg overflow-hidden transform transition-all duration-300 hover:shadow-xl hover:-translate-y-1 cursor-pointer" 
      onClick={handleClick}
      title={`Search for ${rec.name} on Google`}
    >
      <div className="relative aspect-w-2">
        <img
          src={imageUrl}
          alt={`${rec.name} poster`}
          className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300"></div>
      </div>
      <div className="p-4 space-y-3">
        <h4 className="font-semibold text-lg text-gray-900 hover:text-blue-600 line-clamp-2 mb-2 transition-colors duration-200">
          {rec.name}
        </h4>
        <div className="flex flex-wrap gap-1.5 mb-3">
          {rec.genre?.split(',').map((genre, i) => (
            <span 
              key={i} 
              className="px-2.5 py-1 text-xs font-medium rounded-full bg-blue-50 text-blue-600 hover:bg-blue-100 transition-colors duration-200"
              onClick={(e) => e.stopPropagation()} // Prevent triggering parent's onClick
            >
              {genre.trim()}
            </span>
          ))}
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-yellow-400">★</span>
            <span className="text-sm font-semibold text-gray-700">
              {rec.rating?.toFixed(2) || 'N/A'}
            </span>
          </div>
          <div className="text-xs text-blue-600 flex items-center space-x-1">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
            <span>Search on Google</span>
          </div>
        </div>
      </div>
    </div>
  );
};

const GenreRecommendationPage = ({ apiUrl }) => {
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [topN, setTopN] = useState(5);
  const [ratingThreshold, setRatingThreshold] = useState(6.0);
  const [recommendations, setRecommendations] = useState([]);
  const [readyCards, setReadyCards] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedType, setSelectedType] = useState('ALL');

  // Available genres list
  const availableGenres = [
    'Action', 'Adventure', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama',
    'Ecchi', 'Fantasy', 'Game', 'Harem', 'Hentai', 'Historical', 'Horror',
    'Josei', 'Kids', 'Magic', 'Martial Arts', 'Mecha', 'Military', 'Music',
    'Mystery', 'Parody','Psychological', 'Romance',
    'School', 'Sci-Fi', 'Seinen', 'Shoujo', 'Shounen' 
    , 'Slice of Life', 'Space', 'Sports', 'Super Power',
     'Thriller', 'Yaoi'
  ];

  // Available types list
  const availableTypes = ['ALL', 'Movie', 'Music', 'ONA', 'OVA', 'Special', 'TV'];

  // Load cached images on component mount
  useEffect(() => {
    try {
      const cachedData = localStorage.getItem('animeImageCache');
      if (cachedData) {
        const { images: cachedImages, timestamp } = JSON.parse(cachedData);
        if (Date.now() - timestamp < CACHE_EXPIRY) {
          setReadyCards(Object.entries(cachedImages).map(([name, url]) => ({ name, url })));
        } else {
          localStorage.removeItem('animeImageCache');
        }
      }
    } catch (err) {
      console.error('Error loading cached images:', err);
    }
  }, []);

  // Save images to cache whenever they update
  useEffect(() => {
    if (Object.keys(readyCards).length > 0) {
      try {
        localStorage.setItem('animeImageCache', JSON.stringify({
          images: readyCards.reduce((acc, card) => ({ ...acc, [card.name]: card.url }), {}),
          timestamp: Date.now()
        }));
      } catch (err) {
        console.error('Error caching images:', err);
      }
    }
  }, [readyCards]);

  const cleanAnimeName = (name) => {
    return name
      .replace(/[:\-～!@#$%^&*()_+=]/g, ' ')
      .replace(/\s+/g, ' ')
      .replace(/(TV|Movie|OVA|Special)$/i, '')
      .replace(/Season \d+/i, '')
      .replace(/Part \d+/i, '')
      .trim();
  };

  const fetchImageWithRetry = async (animeName, retryCount = 3) => {
    // Check cache first
    const cachedData = localStorage.getItem('animeImageCache');
    if (cachedData) {
      const { images: cachedImages, timestamp } = JSON.parse(cachedData);
      if (Date.now() - timestamp < CACHE_EXPIRY && cachedImages[animeName]) {
        return cachedImages[animeName];
      }
    }

    const searchVariations = [
      animeName,
      cleanAnimeName(animeName),
      animeName.split(':')[0],
      cleanAnimeName(animeName).split(' ')[0]
    ];

    for (let variation of searchVariations) {
      for (let attempt = 0; attempt < retryCount; attempt++) {
        try {
          const searchResponse = await fetch(
            `https://api.jikan.moe/v4/anime?q=${encodeURIComponent(variation)}&limit=5`
          );

          if (!searchResponse.ok) {
            if (searchResponse.status === 429) {
              await new Promise(resolve => setTimeout(resolve, 1000));
              continue;
            }
            throw new Error(`Search failed: ${searchResponse.status}`);
          }

          const searchData = await searchResponse.json();
          const results = searchData.data || [];

          const bestMatch = results.find(anime => {
            const cleanTitle = cleanAnimeName(anime.title).toLowerCase();
            const cleanSearchTitle = cleanAnimeName(variation).toLowerCase();
            return cleanTitle.includes(cleanSearchTitle) || 
                   cleanSearchTitle.includes(cleanTitle) ||
                   (anime.titles || []).some(t => 
                     cleanAnimeName(t.title).toLowerCase().includes(cleanSearchTitle)
                   );
          });

          if (bestMatch?.images?.jpg?.image_url) {
            const img = new Image();
            await new Promise((resolve, reject) => {
              img.onload = resolve;
              img.onerror = reject;
              img.src = bestMatch.images.jpg.image_url;
            });
            return bestMatch.images.jpg.image_url;
          }
        } catch (err) {
          console.error(`Attempt ${attempt + 1} failed for ${variation}:`, err);
          if (attempt === retryCount - 1) continue;
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
    }
    throw new Error('No matching image found');
  };

  const handleGenreToggle = (genre) => {
    setSelectedGenres(prev => 
      prev.includes(genre)
        ? prev.filter(g => g !== genre)
        : [...prev, genre]
    );
  };

  const fetchRecommendations = async () => {
    if (selectedGenres.length === 0) {
      setError('Please select at least one genre');
      return;
    }

    setLoading(true);
    setError(null);
    setRecommendations([]);
    setReadyCards([]);
    
    try {
      console.log('Fetching recommendations for genres:', selectedGenres);
      
      const response = await fetch(
        `${apiUrl}/recommend/genre`,
        {
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            genres: selectedGenres,
            type_anime: selectedType.toLowerCase(),
            top_n: topN
          })
        }
      );
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }
      
      const data = await response.json();
      console.log('API Response:', data);
      
      // Combine popular and relevant recommendations
      const allRecommendations = [...(data.popular || []), ...(data.relevant || [])];
      
      if (allRecommendations.length > 0) {
        // Filter by rating threshold and limit to topN
        const filteredRecs = allRecommendations
          .filter(rec => rec.rating >= ratingThreshold)
          .slice(0, topN);
        
        if (filteredRecs.length > 0) {
          setRecommendations(filteredRecs);
          await fetchImages(filteredRecs);
        } else {
          setError('No recommendations found meeting the rating threshold');
          setLoading(false);
        }
      } else if (data.message) {
        setError(data.message);
        setLoading(false);
      } else {
        setError('No recommendations found for these genres');
        setLoading(false);
      }
    } catch (error) {
      console.error('Fetch error:', error);
      setError('Error fetching recommendations: ' + error.message);
      setLoading(false);
    }
  };

  const fetchImages = async (recs) => {
    const newReadyCards = [];
    
    for (let i = 0; i < recs.length; i += BATCH_SIZE) {
      const batch = recs.slice(i, i + BATCH_SIZE);
      
      const batchPromises = batch.map(async (rec) => {
        try {
          const imageUrl = await fetchImageWithRetry(rec.name);
          if (imageUrl) {
            newReadyCards.push({ rec, imageUrl });
            setReadyCards(current => [...current, { rec, imageUrl }]);
          }
        } catch (err) {
          console.error(`Error fetching image for ${rec.name}:`, err);
        }
      });

      await Promise.all(batchPromises);
      
      if (i === 0) {
        setLoading(false);
      }
      
      if (i + BATCH_SIZE < recs.length) {
        await new Promise(resolve => setTimeout(resolve, RATE_LIMIT_DELAY));
      }
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h2 className="text-3xl font-bold text-gray-900 mb-8">Find Anime by Genres</h2>
      
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <div className="max-w-2xl space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Genres
            </label>
            <div className="flex flex-wrap gap-2">
              {availableGenres.map(genre => (
                <button
                  key={genre}
                  onClick={() => handleGenreToggle(genre)}
                  className={`px-4 py-2 rounded-full text-sm font-medium transition-colors duration-200 ${
                    selectedGenres.includes(genre)
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                  disabled={loading}
                >
                  {genre}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Type
            </label>
            <div className="flex flex-wrap gap-2">
              {availableTypes.map(type => (
                <button
                  key={type}
                  onClick={() => setSelectedType(type)}
                  className={`px-4 py-2 rounded-full text-sm font-medium transition-colors duration-200 ${
                    selectedType === type
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                  disabled={loading}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Number of Results
              </label>
              <input
                type="number"
                value={topN}
                onChange={(e) => setTopN(Math.max(1, Math.min(200, parseInt(e.target.value))))}
                min="1"
                max="200"
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors duration-200"
                disabled={loading}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Minimum Rating
              </label>
              <input
                type="number"
                value={ratingThreshold}
                onChange={(e) => setRatingThreshold(Math.max(0, Math.min(10, parseFloat(e.target.value))))}
                min="0"
                max="10"
                step="0.1"
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors duration-200"
                disabled={loading}
              />
            </div>
          </div>

          <div>
            <button
              onClick={fetchRecommendations}
              disabled={loading || selectedGenres.length === 0}
              className={`w-full sm:w-auto px-6 py-3 bg-blue-600 text-white font-medium rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors duration-200 ${
                loading ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Searching...
                </span>
              ) : (
                'Find Recommendations'
              )}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-8 rounded-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          </div>
        </div>
      )}

      {loading && <LoadingSpinner />}

      {!loading && recommendations.length > 0 && (
        <div>
          <h3 className="text-2xl font-semibold text-gray-900 mb-6">
            Recommended Anime for {selectedGenres.join(', ')}
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6">
            {recommendations.map((rec, index) => {
              const readyCard = readyCards.find(card => card.rec.name === rec.name);
              return readyCard ? (
                <AnimeCard key={index} rec={readyCard.rec} imageUrl={readyCard.imageUrl} />
              ) : (
                <LoadingCard key={index} />
              );
            })}
          </div>
        </div>
      )}

      {!loading && !error && recommendations.length === 0 && (
        <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-blue-700">Select one or more genres to get anime recommendations.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GenreRecommendationPage;