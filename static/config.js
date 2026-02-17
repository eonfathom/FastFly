// Configuration for frontend deployment
// Add this near the top of your HTML files or create as a separate config.js

// API endpoint configuration
const API_CONFIG = {
    // Will use environment variable if available, otherwise fallback to localhost
    baseUrl: typeof VITE_API_URL !== 'undefined' 
        ? VITE_API_URL 
        : 'http://localhost:8000',
    
    // WebSocket URL (ws:// or wss:// depending on http:// or https://)
    get wsUrl() {
        return this.baseUrl.replace(/^http/, 'ws');
    }
};

// Helper function to get API endpoint
function getApiUrl(path) {
    return `${API_CONFIG.baseUrl}${path}`;
}

// Helper function to get WebSocket URL
function getWsUrl(path) {
    return `${API_CONFIG.wsUrl}${path}`;
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { API_CONFIG, getApiUrl, getWsUrl };
}
