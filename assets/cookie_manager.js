// assets/cookie_manager.js
// Cookie management for user token persistence

class CookieManager {
    constructor() {
        this.cookieName = 'avocado_user_token';
        this.cookieExpiry = 365; // days
    }

    // Set cookie with expiration
    setCookie(value) {
        const date = new Date();
        date.setTime(date.getTime() + (this.cookieExpiry * 24 * 60 * 60 * 1000));
        const expires = "expires=" + date.toUTCString();
        document.cookie = `${this.cookieName}=${value};${expires};path=/;SameSite=Strict`;
        
        // Trigger custom event to notify Dash
        window.dispatchEvent(new CustomEvent('cookieSet', {
            detail: { token: value }
        }));
    }

    // Get cookie value
    getCookie() {
        const name = this.cookieName + "=";
        const decodedCookie = decodeURIComponent(document.cookie);
        const cookieArray = decodedCookie.split(';');
        
        for (let i = 0; i < cookieArray.length; i++) {
            let cookie = cookieArray[i];
            while (cookie.charAt(0) === ' ') {
                cookie = cookie.substring(1);
            }
            if (cookie.indexOf(name) === 0) {
                return cookie.substring(name.length, cookie.length);
            }
        }
        return null;
    }

    // Generate new UUID token
    generateToken() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Get or create user token
    getOrCreateToken() {
        let token = this.getCookie();
        if (!token) {
            token = this.generateToken();
            this.setCookie(token);
        }
        return token;
    }

    // Clear cookie (for logout functionality)
    clearCookie() {
        document.cookie = `${this.cookieName}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;`;
        
        // Trigger custom event
        window.dispatchEvent(new CustomEvent('cookieCleared'));
    }
}

// Initialize cookie manager when DOM is loaded
let cookieManager;

document.addEventListener('DOMContentLoaded', function() {
    cookieManager = new CookieManager();
    
    // Set up interval to check and update token in Dash
    setInterval(function() {
        const token = cookieManager.getOrCreateToken();
        
        // Update hidden div in Dash if it exists
        const tokenStore = document.getElementById('user-token-store');
        if (tokenStore && tokenStore.textContent !== token) {
            tokenStore.textContent = token;
            
            // Trigger change event for Dash to pick up
            const event = new Event('change', { bubbles: true });
            tokenStore.dispatchEvent(event);
        }
    }, 1000); // Check every second
});

// Expose globally for Dash callbacks
window.cookieManager = cookieManager;

// Custom Dash component for cookie integration
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    cookieManager: {
        // Get current user token
        get_user_token: function() {
            if (window.cookieManager) {
                return window.cookieManager.getOrCreateToken();
            }
            return null;
        },
        
        // Set user token (useful for login scenarios)
        set_user_token: function(token) {
            if (window.cookieManager && token) {
                window.cookieManager.setCookie(token);
                return token;
            }
            return null;
        },
        
        // Clear user session
        clear_user_session: function() {
            if (window.cookieManager) {
                window.cookieManager.clearCookie();
                return null;
            }
            return null;
        }
    }
});

// Additional utility functions for enhanced UX

// File upload drag and drop enhancement
function enhanceFileUpload() {
    const uploadAreas = document.querySelectorAll('[id*="upload"]');
    
    uploadAreas.forEach(area => {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            area.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, unhighlight, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        function highlight(e) {
            area.style.borderColor = '#81a1c1'; // Nord frost blue
            area.style.backgroundColor = '#81a1c120'; // Semi-transparent
        }

        function unhighlight(e) {
            area.style.borderColor = '#4c566a'; // Back to original
            area.style.backgroundColor = '#434c5e'; // Back to original
        }
    });
}

// Initialize file upload enhancements after page load
document.addEventListener('DOMContentLoaded', function() {
    // Delay to ensure Dash components are rendered
    setTimeout(enhanceFileUpload, 2000);
    
    // Re-apply after tab changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                setTimeout(enhanceFileUpload, 500);
            }
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});

// Form validation enhancements
function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

function validateAlgorithmName(name) {
    // Check for reasonable length and no special characters that could cause issues
    const nameRegex = /^[a-zA-Z0-9\s\-\._()]+$/;
    return name && name.length >= 3 && name.length <= 100 && nameRegex.test(name);
}

// Export validation functions for use in Dash
window.formValidation = {
    validateEmail: validateEmail,
    validateAlgorithmName: validateAlgorithmName
};

// Performance monitoring for file uploads
class UploadMonitor {
    constructor() {
        this.uploads = new Map();
    }
    
    startUpload(uploadId, filename, fileSize) {
        this.uploads.set(uploadId, {
            filename: filename,
            fileSize: fileSize,
            startTime: Date.now(),
            progress: 0
        });
    }
    
    updateProgress(uploadId, progress) {
        const upload = this.uploads.get(uploadId);
        if (upload) {
            upload.progress = progress;
            upload.currentTime = Date.now();
            
            // Calculate speed and ETA
            const elapsed = (upload.currentTime - upload.startTime) / 1000; // seconds
            const speed = (upload.fileSize * progress) / elapsed; // bytes per second
            const remaining = upload.fileSize * (1 - progress);
            const eta = remaining / speed; // seconds
            
            return {
                progress: progress,
                speed: this.formatBytes(speed) + '/s',
                eta: this.formatTime(eta),
                elapsed: this.formatTime(elapsed)
            };
        }
        return null;
    }
    
    finishUpload(uploadId) {
        const upload = this.uploads.get(uploadId);
        if (upload) {
            const totalTime = (Date.now() - upload.startTime) / 1000;
            const avgSpeed = upload.fileSize / totalTime;
            
            this.uploads.delete(uploadId);
            
            return {
                success: true,
                totalTime: this.formatTime(totalTime),
                avgSpeed: this.formatBytes(avgSpeed) + '/s'
            };
        }
        return null;
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    formatTime(seconds) {
        if (seconds < 60) return Math.round(seconds) + 's';
        if (seconds < 3600) return Math.round(seconds / 60) + 'm ' + Math.round(seconds % 60) + 's';
        return Math.round(seconds / 3600) + 'h ' + Math.round((seconds % 3600) / 60) + 'm';
    }
}

window.uploadMonitor = new UploadMonitor();

// Theme switcher (future enhancement)
class ThemeManager {
    constructor() {
        this.themes = {
            nord: 'nord-theme',
            light: 'light-theme',
            dark: 'dark-theme'
        };
        this.currentTheme = 'nord';
    }
    
    switchTheme(themeName) {
        if (this.themes[themeName]) {
            document.body.className = document.body.className.replace(/theme-\w+/g, '');
            document.body.classList.add(`theme-${themeName}`);
            this.currentTheme = themeName;
            
            // Save to localStorage
            localStorage.setItem('avocado_theme', themeName);
        }
    }
    
    loadSavedTheme() {
        const savedTheme = localStorage.getItem('avocado_theme');
        if (savedTheme && this.themes[savedTheme]) {
            this.switchTheme(savedTheme);
        }
    }
}

window.themeManager = new ThemeManager();

// Load saved theme on page load
document.addEventListener('DOMContentLoaded', function() {
    window.themeManager.loadSavedTheme();
});