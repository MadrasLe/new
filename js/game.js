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

// Create a pane for countries so they sit above tiles but below markers
map.createPane('countriesPane');
map.getPane('countriesPane').style.zIndex = 400;

console.log("Map loaded");

// --- Load Country Borders ---
let geojsonLayer;

fetch('js/countries.geo.json')
    .then(response => response.json())
    .then(data => {
        geojsonLayer = L.geoJson(data, {
            pane: 'countriesPane',
            style: styleCountry,
            onEachFeature: onEachCountry
        }).addTo(map);
    })
    .catch(err => console.error("Error loading GeoJSON:", err));

function styleCountry(feature) {
    return {
        fillColor: getColor(feature.properties.name),
        weight: 1,
        opacity: 1,
        color: '#444', // Border color
        dashArray: '3',
        fillOpacity: 0.1 // Low opacity to see the dark map behind, highlights on hover
    };
}

function getColor(name) {
    // Simple hash to give consistent random-ish colors to countries for "factions"
    let hash = 0;
    for (let i = 0; i < name.length; i++) {
        hash = name.charCodeAt(i) + ((hash << 5) - hash);
    }
    const c = (hash & 0x00FFFFFF).toString(16).toUpperCase();
    return '#' + "00000".substring(0, 6 - c.length) + c;
}

function highlightFeature(e) {
    var layer = e.target;

    layer.setStyle({
        weight: 2,
        color: '#666',
        dashArray: '',
        fillOpacity: 0.3
    });

    if (!L.Browser.ie && !L.Browser.opera && !L.Browser.edge) {
        layer.bringToFront();
    }
}

function resetHighlight(e) {
    geojsonLayer.resetStyle(e.target);
}

function zoomToFeature(e) {
    map.fitBounds(e.target.getBounds());
    selectEntity({
        name: e.target.feature.properties.name,
        type: "Nation",
        description: `Territory of ${e.target.feature.properties.name}.`
    });
}

function onEachCountry(feature, layer) {
    layer.on({
        mouseover: highlightFeature,
        mouseout: resetHighlight,
        click: zoomToFeature
    });
}

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
    // Attach custom property to marker to identify it
    marker.gameData = unit;

    marker.on('click', (e) => {
        L.DomEvent.stopPropagation(e); // Prevent map click
        selectEntity(unit);
        gameState.selectedMarker = marker; // Track actual marker for movement
    });
});

// --- Unit Movement Logic ---
// Right-click on map to move selected unit
map.on('contextmenu', (e) => {
    if (gameState.selectedMarker) {
        const destination = e.latlng;

        // Simple animation: set new LatLng
        // For a game, you'd want a pathfinding or travel time logic.
        // Here we just "teleport" or "move" directly for prototype.

        console.log(`Moving ${gameState.selectedMarker.gameData.name} to`, destination);

        // Create a line to show path temporarily
        const pathLine = L.polyline([gameState.selectedMarker.getLatLng(), destination], {
            color: 'cyan',
            dashArray: '5, 10',
            weight: 2
        }).addTo(map);

        // Simulate travel time (1 second)
        setTimeout(() => {
            gameState.selectedMarker.setLatLng(destination);
            gameState.selectedMarker.gameData.lat = destination.lat;
            gameState.selectedMarker.gameData.lng = destination.lng;
            map.removeLayer(pathLine);
        }, 1000);
    }
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

    // Dynamic content based on type
    let details = "";
    if (entity.type === "Nation") {
        // Mock data for countries
        const gdp = Math.floor(Math.random() * 1000) + 10;
        const population = Math.floor(Math.random() * 100) + 1;
        details = `
            <strong>Status:</strong> Sovereign Nation<br>
            <strong>Population:</strong> ${population} Million<br>
            <strong>GDP:</strong> $${gdp} Billion<br>
            <strong>Factions:</strong> Non-Aligned<br>
            <br>
            <em>${entity.description}</em>
        `;
    } else {
        // Unit data
        details = `
            <strong>Type:</strong> ${entity.type}<br>
            <strong>Strength:</strong> 100%<br>
            <strong>Supply:</strong> 85%<br>
            <br>
            ${entity.description ? entity.description : "Unit awaiting orders. Right-click map to move."}
        `;
    }

    content.innerHTML = details;
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
