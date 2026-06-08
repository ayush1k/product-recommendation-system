import React, { useState, useEffect } from 'react';

const App = () => {
  const [products, setProducts] = useState([]);
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch all products on initial mount
  useEffect(() => {
    fetchAllProducts();
  }, []);

  // Dynamic API URL detection for Codespaces vs Local
  const getApiUrl = () => {
    const hostname = window.location.hostname;
    if (hostname.endsWith('.github.dev') || hostname.endsWith('.gitpod.io')) {
      // For cloud IDEs, construct the backend URL by replacing the frontend port with the backend port
      return `https://${hostname.replace('-5173.', '-8000.')}/api`;
    }
    // Default for local development
    return 'http://localhost:8000/api';
  };

  // Use relative `/api` during Vite dev to allow the dev-server proxy to forward requests to the backend.
  // In production or unknown hosts, fall back to the dynamic URL detection.
  const API_BASE = import.meta.env?.DEV ? '/api' : getApiUrl();

  const fetchAllProducts = async () => {
    setIsInitialLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/products`);
      if (!response.ok) throw new Error('Failed to load products');
      const data = await response.json();
      setProducts(data);
    } catch (err) {
      setError('Could not connect to the backend server.');
    } finally {
      setIsInitialLoading(false);
    }
  };

  const handleRecommend = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) throw new Error('Recommendation failed');
      
      const data = await response.json();
      setProducts(data);
    } catch (err) {
      setError('The AI service is currently unavailable. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setQuery('');
    fetchAllProducts();
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col font-sans">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 py-4 px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
              </svg>
            </div>
            <h1 className="text-2xl font-extrabold text-gray-900 tracking-tight">ProductPulse <span className="text-indigo-600">AI</span></h1>
          </div>
          <div className="hidden md:block text-gray-500 text-sm font-medium">
            AI-Driven Shopping Experience
          </div>
        </div>
      </header>

      <main className="flex-grow max-w-6xl mx-auto w-full px-6 py-10">
        {/* Search Section */}
        <section className="bg-white rounded-2xl shadow-xl p-8 mb-12 border border-gray-100 transition-all">
          <div className="max-w-3xl">
            <h2 className="text-xl font-bold text-gray-800 mb-2">Find your perfect match</h2>
            <p className="text-gray-500 mb-6">Describe what you're looking for, and our AI will find the best options from our catalog.</p>
            
            <form onSubmit={handleRecommend} className="flex flex-col md:flex-row gap-3">
              <div className="flex-grow relative">
                <input
                  type="text"
                  placeholder='e.g., "I need a budget phone for my kid" or "high-end audio for travel"'
                  className="w-full px-5 py-4 rounded-xl border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all pr-12 text-gray-700"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <div className="absolute right-4 top-4 text-gray-400">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={isLoading || !query.trim()}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold px-8 py-4 rounded-xl transition-all shadow-lg shadow-indigo-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center min-w-[180px]"
                >
                  {isLoading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin h-5 w-5 mr-3 text-white" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Thinking...
                    </span>
                  ) : 'Find Products'}
                </button>
                <button
                  type="button"
                  onClick={handleReset}
                  className="px-6 py-4 bg-gray-100 hover:bg-gray-200 text-gray-600 font-bold rounded-xl transition-all border border-gray-200"
                >
                  Reset
                </button>
              </div>
            </form>
          </div>
        </section>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8 rounded-r-lg animate-pulse">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700 font-medium">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Product Grid */}
        <section>
          <div className="flex justify-between items-end mb-8">
            <div>
              <h3 className="text-2xl font-bold text-gray-900">
                {query && !isLoading ? 'Personalized Recommendations' : 'Product Catalog'}
              </h3>
              <p className="text-gray-500">Showing {products.length} items</p>
            </div>
          </div>

          {isInitialLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
              {[...Array(8)].map((_, i) => (
                <div key={i} className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100 animate-pulse">
                  <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
                  <div className="h-6 bg-gray-200 rounded w-3/4 mb-4"></div>
                  <div className="h-20 bg-gray-200 rounded mb-4"></div>
                  <div className="h-8 bg-gray-200 rounded w-full"></div>
                </div>
              ))}
            </div>
          ) : products.length === 0 ? (
            <div className="text-center py-24 bg-white rounded-3xl border-2 border-dashed border-gray-200 shadow-inner">
              <div className="mx-auto w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mb-6">
                <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.172 9.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h4 className="text-xl font-bold text-gray-800 mb-2">No products found</h4>
              <p className="text-gray-500 max-w-xs mx-auto mb-8">We couldn't find anything matching your request. Try broadening your search.</p>
              <button onClick={handleReset} className="text-indigo-600 font-bold hover:underline">View all products</button>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
              {products.map((product) => (
                <div key={product.id} className="group bg-white rounded-2xl shadow-sm hover:shadow-2xl transition-all duration-300 border border-gray-100 flex flex-col overflow-hidden hover:-translate-y-1">
                  <div className="p-6 flex-grow">
                    <div className="flex justify-between items-start mb-4">
                      <span className="text-[10px] font-black px-3 py-1 bg-indigo-50 text-indigo-700 rounded-full uppercase tracking-tighter">
                        {product.category}
                      </span>
                      <span className="text-xl font-black text-gray-900">
                        ${product.price.toLocaleString()}
                      </span>
                    </div>
                    <h4 className="text-lg font-bold text-gray-800 mb-3 group-hover:text-indigo-600 transition-colors">
                      {product.name}
                    </h4>
                    <p className="text-gray-500 text-sm leading-relaxed line-clamp-4">
                      {product.description}
                    </p>
                  </div>
                  <div className="px-6 pb-6 pt-0">
                    <button className="w-full py-3 bg-gray-900 text-white rounded-xl text-sm font-bold hover:bg-indigo-600 transition-colors shadow-lg shadow-gray-200">
                      Add to Cart
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>

      <footer className="bg-white border-t border-gray-200 py-10 mt-20">
        <div className="max-w-6xl mx-auto px-6 text-center">
          <p className="text-gray-400 text-sm">© 2026 ProductPulse AI. Powered by FastAPI, LangChain & React.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
