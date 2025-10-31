let analysisData = null;
let speedometerValue = 0;

function analyzeRace() {
    // Get form values
    const conditions = {
        weather: document.getElementById('weather').value,
        trackType: document.getElementById('trackType').value,
        waviness: document.getElementById('waviness').value,
        trackLength: document.getElementById('trackLength').value,
        curves: document.getElementById('curves').value,
        raceDuration: document.getElementById('raceDuration').value
    };

    // Show loading screen
    document.getElementById('inputForm').classList.add('hidden');
    document.getElementById('loadingScreen').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');

    // Send request to Flask backend
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(conditions)
    })
    .then(response => response.json())
    .then(data => {
        analysisData = data;
        setTimeout(() => {
            displayResults(data, conditions);
        }, 2000);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during analysis. Please try again.');
        resetToInput();
    });
}

function displayResults(data, conditions) {
    document.getElementById('loadingScreen').classList.add('hidden');
    document.getElementById('results').classList.remove('hidden');
    
    const recommended = data.recommended;
    const alternatives = data.alternatives;
    const allCars = data.allCars;
    
    // Animate speedometer
    animateSpeedometer(recommended.car.topSpeed);
    
    // Build results HTML
    const resultsHTML = `
        <!-- Recommended Car -->
        <div class="bg-gradient-to-br from-red-900/50 to-gray-900 rounded-2xl p-8 border-2 border-red-500 shadow-2xl shadow-red-500/30 animate-fadeIn">
            <div class="flex items-center justify-between mb-6">
                <h2 class="text-4xl font-bold flex items-center gap-3">
                    <svg class="w-10 h-10 text-yellow-400" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                    </svg>
                    Optimal Car Selection
                </h2>
                <div class="text-6xl">${recommended.car.image}</div>
            </div>
            
            <div class="grid md:grid-cols-2 gap-8">
                <div>
                    <h3 class="text-3xl font-bold text-red-400 mb-2">${recommended.name}</h3>
                    <p class="text-xl text-gray-300 mb-4">${recommended.car.team} • ${recommended.car.year}</p>
                    <p class="text-lg text-gray-400 mb-4">Power Unit: ${recommended.car.powerUnit}</p>
                    
                    <div class="bg-black/50 rounded-lg p-4 mb-4">
                        <div class="text-sm text-gray-400 mb-1">Performance Score</div>
                        <div class="text-5xl font-bold text-transparent bg-gradient-to-r from-yellow-400 to-red-500 bg-clip-text">
                            ${Math.round(recommended.score)}
                        </div>
                    </div>

                    <div class="space-y-3">
                        <h4 class="text-lg font-semibold text-red-400">Key Factors:</h4>
                        ${recommended.factors.slice(0, 3).map(factor => `
                            <div class="bg-black/30 rounded-lg p-3">
                                <div class="flex justify-between items-center mb-2">
                                    <span class="text-gray-300">${factor.name}</span>
                                    <span class="text-red-400 font-bold">${Math.round(factor.value)}</span>
                                </div>
                                <div class="w-full bg-gray-700 rounded-full h-2">
                                    <div class="bg-gradient-to-r from-red-500 to-yellow-400 h-2 rounded-full transition-all duration-1000" 
                                        style="width: ${factor.value}%"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div>
                    <div id="speedometer" class="relative w-64 h-32 mx-auto mb-6">
                        <svg viewBox="0 0 200 100" class="w-full h-full">
                            <defs>
                                <linearGradient id="speedGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stop-color="#dc2626" />
                                    <stop offset="50%" stop-color="#fbbf24" />
                                    <stop offset="100%" stop-color="#10b981" />
                                </linearGradient>
                            </defs>
                            <path d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="#374151" stroke-width="15" />
                            <path id="speedArc" d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="url(#speedGradient)" 
                                stroke-width="15" stroke-dasharray="251.2" stroke-dashoffset="251.2" 
                                style="transition: stroke-dashoffset 1s ease-out" />
                            <line id="speedNeedle" x1="100" y1="90" x2="100" y2="30" stroke="#ef4444" stroke-width="3"
                                transform="rotate(-90 100 90)" style="transition: transform 1s ease-out" />
                            <circle cx="100" cy="90" r="8" fill="#ef4444" />
                        </svg>
                        <div class="absolute bottom-0 left-0 right-0 text-center">
                            <div id="speedValue" class="text-4xl font-bold text-red-500">0</div>
                            <div class="text-sm text-gray-400">km/h</div>
                        </div>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-4">
                        ${[
                            { label: 'Cornering', value: recommended.car.cornering, icon: '🔄' },
                            { label: 'Acceleration', value: recommended.car.acceleration, icon: '⚡' },
                            { label: 'Reliability', value: recommended.car.reliability, icon: '🔧' },
                            { label: 'Aero', value: recommended.car.aero, icon: '✈' }
                        ].map(stat => `
                            <div class="bg-black/40 rounded-lg p-3 text-center">
                                <div class="text-2xl mb-1">${stat.icon}</div>
                                <div class="text-2xl font-bold text-red-400">${stat.value}</div>
                                <div class="text-xs text-gray-400">${stat.label}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;

    document.getElementById('results').innerHTML = resultsHTML;
}

function animateSpeedometer(maxSpeed) {
    let currentSpeed = 0;
    const increment = maxSpeed / 50;
    
    const interval = setInterval(() => {
        currentSpeed += increment;
        if (currentSpeed >= maxSpeed) {
            currentSpeed = maxSpeed;
            clearInterval(interval);
        }
        
        const speed = Math.floor(currentSpeed);
        const percentage = (speed / maxSpeed) * 100;
        const rotation = (percentage / 100) * 180 - 90;
        
        document.getElementById('speedValue').textContent = speed;
        document.getElementById('speedArc').style.strokeDashoffset = 251.2 - (251.2 * percentage) / 100;
        document.getElementById('speedNeedle').setAttribute('transform', `rotate(${rotation} 100 90)`);
    }, 30);
}

function resetToInput() {
    document.getElementById('results').classList.add('hidden');
    document.getElementById('inputForm').classList.remove('hidden');
}
