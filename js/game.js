// Initialize map
// Centered on Brazil/South America for context, zoom level 3
const map = L.map('map', {
    zoomControl: false // We can move it or custom style it
}).setView([-14.2350, -51.9253], 4);

// Add zoom control to top right
L.control.zoom({
    position: 'bottomright'
}).addTo(map);

// Dark Theme Tiles (CartoDB Dark Matter)
L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19
}).addTo(map);

console.log("Map loaded");

// --- Game State ---
let gameState = {
    money: 1000000,
    fuel: 5000,
    manpower: 25000,
    date: new Date("2024-01-01"),
    selectedEntity: null
};

// --- UI Updaters ---
function updateUI() {
    document.getElementById('money').innerText = gameState.money.toLocaleString();
    document.getElementById('fuel').innerText = gameState.fuel.toLocaleString();
    document.getElementById('manpower').innerText = gameState.manpower.toLocaleString();
    document.getElementById('game-date').innerText = gameState.date.toISOString().split('T')[0];
}

// --- Game Loop (Time Progression) ---
setInterval(() => {
    // Advance time by 1 day every second
    gameState.date.setDate(gameState.date.getDate() + 1);

    // Passive income calculation
    gameState.money += 100;
    gameState.fuel += 10;

    updateUI();
}, 1000);

// --- Entities (Units/Territories) ---
// Define a custom icon for units
const tankIcon = L.divIcon({
    className: 'custom-div-icon',
    html: "<div style='background-color:cyan;width:10px;height:10px;border-radius:50%;border:2px solid white;'></div>",
    iconSize: [14, 14],
    iconAnchor: [7, 7]
});

// Example Units
const units = [
    { id: 1, name: "1st Armored Division", lat: -23.5505, lng: -46.6333, type: "Tank" }, // Sao Paulo
    { id: 2, name: "2nd Infantry Brigade", lat: -22.9068, lng: -43.1729, type: "Infantry" } // Rio
];

units.forEach(unit => {
    const marker = L.marker([unit.lat, unit.lng], {icon: tankIcon}).addTo(map);

    marker.on('click', () => {
        selectEntity(unit);
    });
});

// Example Territory (Polygon)
const territoryCoords = [
    [-15.7, -48.0],
    [-15.7, -47.8],
    [-15.9, -47.8],
    [-15.9, -48.0]
]; // Approx Brasilia area

const territory = L.polygon(territoryCoords, {color: '#4ca3dd', fillColor: '#4ca3dd', fillOpacity: 0.3}).addTo(map);

territory.on('click', () => {
    selectEntity({
        name: "Brasília Defense Zone",
        type: "Territory",
        description: "Capital region. Strategic importance: High."
    });
});

// --- Selection Logic ---
function selectEntity(entity) {
    gameState.selectedEntity = entity;

    const panel = document.getElementById('info-panel');
    const title = document.getElementById('panel-title');
    const content = document.getElementById('panel-content');

    panel.classList.remove('hidden');
    title.innerText = entity.name;
    content.innerHTML = `
        <strong>Type:</strong> ${entity.type}<br>
        ${entity.description ? entity.description : "Unit ready for orders."}
    `;

    // Highlight effect logic could go here
    console.log("Selected:", entity.name);
}

// Click on map background to deselect
map.on('click', (e) => {
    // Check if the click target is the map container itself (or the tile layer), not a marker/path
    // Leaflet's map click event fires when you click the map, but we need to ensure we didn't click a unit.
    // However, unit click events propagate. We can use L.DomEvent.stopPropagation on markers if needed,
    // or just rely on the fact that marker click fires first.
    // For simplicity, we won't implement complex deselect logic right now as it might conflict with marker clicks without stopPropagation.
});

updateUI();
