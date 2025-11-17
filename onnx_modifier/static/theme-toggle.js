// Theme Toggle Functionality
(function() {
    'use strict';
    
    // Load saved theme immediately (before DOMContentLoaded to prevent flash)
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.documentElement.classList.add('dark-theme');
        if (document.body) {
            document.body.classList.add('dark-theme');
        }
    }
    
    // Theme Toggle Function
    function toggleTheme() {
        document.body.classList.toggle('dark-theme');
        const isDark = document.body.classList.contains('dark-theme');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    }
    
    // Attach event listener when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initThemeToggle);
    } else {
        initThemeToggle();
    }
    
    function initThemeToggle() {
        const toggleBtn = document.getElementById('theme-toggle-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', function() {
                toggleTheme();
                updateThemeIcon();
            });
        }
        
        // Apply saved theme to body if not already applied
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark' && !document.body.classList.contains('dark-theme')) {
            document.body.classList.add('dark-theme');
        }
        
        // Update icon on load
        updateThemeIcon();
    }
    
    function updateThemeIcon() {
        const isDark = document.body.classList.contains('dark-theme');
        const lightIcon = document.querySelector('.theme-icon-light');
        const darkIcon = document.querySelector('.theme-icon-dark');
        
        if (lightIcon && darkIcon) {
            if (isDark) {
                lightIcon.style.display = 'none';
                darkIcon.style.display = 'block';
            } else {
                lightIcon.style.display = 'block';
                darkIcon.style.display = 'none';
            }
        }
    }
})();
