// Main application JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Application loaded successfully!');
    
    // Add smooth scrolling for anchor links
    const links = document.querySelectorAll('a[href^="#"]');
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add click animation to features
    const features = document.querySelectorAll('.feature');
    features.forEach(feature => {
        feature.addEventListener('click', function() {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
    
    // Health check function (example API call)
    async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            console.log('Health check:', data);
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }
    
    // Uncomment to perform health check on load
    // checkHealth();
});

// Utility functions
const utils = {
    // Format date
    formatDate: (date) => {
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        }).format(date);
    },
    
    // Debounce function
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Show notification (you can integrate with a toast library)
    showNotification: (message, type = 'info') => {
        console.log(`[${type.toUpperCase()}] ${message}`);
        // Implement your notification system here
    }
};

// Export for use in other scripts
window.AppUtils = utils;
