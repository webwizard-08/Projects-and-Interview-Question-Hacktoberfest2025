// Command search functionality
function search(platform) {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        alert('Please enter a command to search!'); 
        return;
    }

    // URL mappings
    const platforms = {
        YouTube: `https://www.youtube.com/results?search_query=${encodeURIComponent(query)}`,
        Amazon: `https://www.amazon.in/s?k=${encodeURIComponent(query)}`,
        Spotify: `https://open.spotify.com/search/${encodeURIComponent(query)}`,
        Instagram: `https://www.instagram.com/explore/tags/${encodeURIComponent(query)}/`,
        WhatsApp: `https://wa.me/?text=${encodeURIComponent(query)}`
    };

    // Open the link if platform exists
    if (platforms[platform]) {
        window.open(platforms[platform], '_blank');
    } else {
        alert('Unknown platform!'); 
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll(".holo-btn");
    const hoverSound = document.getElementById("hoverSound");

    buttons.forEach(btn => {
        btn.addEventListener("mouseenter", () => hoverSound.play());
    });
});
