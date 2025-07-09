// bci_communicator/static/bci_communicator/js/communicator.js

class BCICommunicator {
    constructor(config) {
        this.sessionId = config.sessionId;
        this.vocabularyWords = config.vocabularyWords;
        this.rightLetters = config.rightLetters;
        this.leftLetters = config.leftLetters;
        this.csrfToken = config.csrfToken;
        
        // State management
        this.isRunning = false;
        this.pollingInterval = null;
        this.lastEventTimestamp = null;
        this.currentState = 'NAVIGATING';
        
        // UI elements
        this.elements = {};
        
        // Configuration
        this.config = {
            pollingInterval: 500, // ms
            maxEvents: 20,
            animationDuration: 300,
            confidenceThreshold: 0.7
        };
        
        // Bind methods
        this.handleStartClick = this.handleStartClick.bind(this);
        this.handleStopClick = this.handleStopClick.bind(this);
        this.handleLetterClick = this.handleLetterClick.bind(this);
        this.updateInterface = this.updateInterface.bind(this);
    }
    
    init() {
        console.log('Initializing BCI Communicator...');
        this.setupElements();
        this.setupEventListeners();
        this.setupWordSuggestions();
        this.updateCharacterCounts();
        console.log('BCI Communicator initialized');
    }
    
    setupElements() {
        // Get all necessary DOM elements
        this.elements = {
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            currentText: document.getElementById('currentText'),
            currentWord: document.getElementById('currentWord'),
            currentState: document.getElementById('currentState'),
            selectedSide: document.getElementById('selectedSide'),
            currentLetter: document.getElementById('currentLetter'),
            wordCount: document.getElementById('wordCount'),
            charCount: document.getElementById('charCount'),
            motorImageryPredictions: document.getElementById('motorImageryPredictions'),
            wordSuggestions: document.getElementById('wordSuggestions'),
            p300Status: document.getElementById('p300Status'),
            recentEvents: document.getElementById('recentEvents'),
            letterButtons: document.querySelectorAll('.letter-btn'),
            rightLetters: document.querySelectorAll('.right-letter'),
            leftLetters: document.querySelectorAll('.left-letter')
        };
    }
    
    setupEventListeners() {
        // Start/Stop buttons
        this.elements.startBtn.addEventListener('click', this.handleStartClick);
        this.elements.stopBtn.addEventListener('click', this.handleStopClick);
        
        // Letter buttons for manual testing
        this.elements.letterButtons.forEach(btn => {
            btn.addEventListener('click', this.handleLetterClick);
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey) {
                switch(e.key) {
                    case 's':
                        e.preventDefault();
                        this.isRunning ? this.stopCommunication() : this.startCommunication();
                        break;
                    case ' ':
                        e.preventDefault();
                        this.manualAction('add_space');
                        break;
                    case 'c':
                        e.preventDefault();
                        this.manualAction('clear_text');
                        break;
                }
            }
        });
    }
    
    setupWordSuggestions() {
        // Initialize word suggestion display
        this.updateWordSuggestions('');
    }
    
    async handleStartClick() {
        console.log('Starting communication system...');
        await this.startCommunication();
    }
    
    async handleStopClick() {
        console.log('Stopping communication system...');
        await this.stopCommunication();
    }
    
    handleLetterClick(event) {
        const letter = event.target.dataset.letter;
        if (letter) {
            this.manualAction('add_letter', { letter: letter });
        }
    }
    
    async startCommunication() {
        try {
            const response = await this.apiCall(`/start/${this.sessionId}/`, 'POST');
            
            if (response.status === 'success') {
                this.isRunning = true;
                this.elements.startBtn.disabled = true;
                this.elements.stopBtn.disabled = false;
                
                // Start polling for updates
                this.startPolling();
                
                this.showNotification('Communication system started!', 'success');
            } else {
                throw new Error(response.message);
            }
        } catch (error) {
            console.error('Failed to start communication:', error);
            this.showNotification(`Failed to start: ${error.message}`, 'error');
        }
    }
    
    async stopCommunication() {
        try {
            const response = await this.apiCall(`/stop/${this.sessionId}/`, 'POST');
            
            this.isRunning = false;
            this.elements.startBtn.disabled = false;
            this.elements.stopBtn.disabled = true;
            
            // Stop polling
            this.stopPolling();
            
            this.showNotification('Communication system stopped', 'info');
        } catch (error) {
            console.error('Failed to stop communication:', error);
            this.showNotification(`Failed to stop: ${error.message}`, 'error');
        }
    }
    
    startPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        this.pollingInterval = setInterval(() => {
            this.fetchSessionStatus();
        }, this.config.pollingInterval);
        
        // Initial fetch
        this.fetchSessionStatus();
    }
    
    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }
    
    async fetchSessionStatus() {
        try {
            const response = await this.apiCall(`/status/${this.sessionId}/`);
            this.updateInterface(response);
        } catch (error) {
            console.error('Failed to fetch session status:', error);
            // Don't show notifications for polling errors to avoid spam
        }
    }
    
    updateInterface(data) {
        const session = data.session;
        const events = data.events;
        
        // Update session state
        this.updateSessionDisplay(session);
        
        // Process new events
        this.processEvents(events);
        
        // Update predictions if available
        this.updatePredictionDisplay(events);
        
        // Update word suggestions
        this.updateWordSuggestions(session.current_word);
    }
    
    updateSessionDisplay(session) {
        // Update text display
        this.elements.currentText.textContent = session.current_text;
        this.elements.currentWord.textContent = session.current_word;
        
        // Update state indicators
        this.updateStateDisplay(session.communication_state);
        this.elements.selectedSide.textContent = session.selected_side || 'None';
        this.elements.currentLetter.textContent = session.current_letter || 'None';
        
        // Update letter highlighting
        this.updateLetterHighlighting(session);
        
        // Update character counts
        this.updateCharacterCounts();
        
        // Store current state
        this.currentState = session.communication_state;
    }
    
    updateStateDisplay(state) {
        this.elements.currentState.textContent = state;
        this.elements.currentState.className = `badge state-${state.toLowerCase()}`;
        
        // Update state-specific UI
        switch(state) {
            case 'NAVIGATING':
                this.clearLetterHighlighting();
                break;
            case 'SELECTING':
                // Letter highlighting is handled in updateLetterHighlighting
                break;
            case 'CONFIRMING':
                this.showP300Interface();
                break;
        }
    }
    
    updateLetterHighlighting(session) {
        // Clear existing highlighting
        this.clearLetterHighlighting();
        
        if (session.communication_state === 'SELECTING') {
            // Highlight selected side
            const sideClass = session.selected_side === 'RIGHT' ? 'right-letter' : 'left-letter';
            document.querySelectorAll(`.${sideClass}`).forEach(btn => {
                btn.classList.add('selected-side');
            });
            
            // Highlight current letter
            if (session.current_letter) {
                const currentBtn = document.querySelector(`[data-letter="${session.current_letter}"]`);
                if (currentBtn) {
                    currentBtn.classList.add('highlighted');
                }
            }
        }
    }
    
    clearLetterHighlighting() {
        this.elements.letterButtons.forEach(btn => {
            btn.classList.remove('highlighted', 'selected-side');
        });
    }
    
    processEvents(events) {
        // Filter new events
        const newEvents = this.filterNewEvents(events);
        
        // Process each new event
        newEvents.forEach(event => {
            this.processEvent(event);
        });
        
        // Update events display
        this.updateEventsDisplay(events);
    }
    
    filterNewEvents(events) {
        if (!this.lastEventTimestamp) {
            this.lastEventTimestamp = events.length > 0 ? events[0].timestamp : null;
            return events;
        }
        
        const newEvents = events.filter(event => 
            new Date(event.timestamp) > new Date(this.lastEventTimestamp)
        );
        
        if (newEvents.length > 0) {
            this.lastEventTimestamp = newEvents[0].timestamp;
        }
        
        return newEvents;
    }
    
    processEvent(event) {
        switch(event.type) {
            case 'LETTER_SELECTED':
                this.animateLetterSelection(event.selected_letter);
                break;
            case 'WORD_COMPLETED':
                this.animateWordCompletion(event.completed_word);
                break;
            case 'MOTOR_PREDICTION':
                // Handled in updatePredictionDisplay
                break;
            case 'P300_CONFIRMATION':
                this.handleP300Confirmation(event);
                break;
        }
    }
    
    updatePredictionDisplay(events) {
        // Find latest motor imagery prediction
        const latestMIPrediction = events.find(e => e.type === 'MOTOR_PREDICTION');
        
        if (latestMIPrediction && latestMIPrediction.probabilities) {
            this.updateMotorImageryBars(latestMIPrediction.probabilities);
        }
        
        // Find latest P300 prediction
        const latestP300Prediction = events.find(e => e.type === 'P300_CONFIRMATION');
        
        if (latestP300Prediction) {
            this.updateP300Display(latestP300Prediction);
        }
    }
    
    updateMotorImageryBars(probabilities) {
        const bars = this.elements.motorImageryPredictions.querySelectorAll('.confidence-fill');
        const labels = ['Right Hand', 'Left Hand', 'Feet', 'Rest'];
        
        Object.keys(probabilities).forEach((classIdx, i) => {
            if (bars[i]) {
                const confidence = probabilities[classIdx] * 100;
                bars[i].style.width = `${confidence}%`;
                
                // Add high confidence class for animation
                if (confidence > this.config.confidenceThreshold * 100) {
                    bars[i].classList.add('high-confidence');
                } else {
                    bars[i].classList.remove('high-confidence');
                }
                
                // Update label with confidence value
                const container = bars[i].closest('.prediction-container');
                if (container) {
                    const valueSpan = container.querySelector('.prediction-value') || 
                                   this.createPredictionValueSpan(container);
                    valueSpan.textContent = `${confidence.toFixed(1)}%`;
                }
            }
        });
    }
    
    createPredictionValueSpan(container) {
        const label = container.querySelector('.prediction-label');
        const valueSpan = document.createElement('span');
        valueSpan.className = 'prediction-value';
        label.appendChild(valueSpan);
        return valueSpan;
    }
    
    updateP300Display(event) {
        const status = this.elements.p300Status;
        
        if (this.currentState === 'CONFIRMING') {
            status.innerHTML = `
                <div class="small">
                    <strong>Active</strong><br>
                    Confidence: ${(event.confidence * 100).toFixed(1)}%<br>
                    <span class="text-warning">Look at suggested word to confirm</span>
                </div>
            `;
        } else {
            status.innerHTML = '<p class="text-muted small">Inactive</p>';
        }
    }
    
    showP300Interface() {
        // Update P300 status
        this.elements.p300Status.innerHTML = `
            <div class="small">
                <strong class="text-info">P300 Active</strong><br>
                <span class="text-warning">Focus on word suggestion to confirm</span>
            </div>
        `;
    }
    
    updateWordSuggestions(currentWord) {
        const suggestions = this.getWordSuggestions(currentWord);
        
        if (suggestions.length > 0) {
            const suggestionHtml = suggestions.map(word => 
                `<button class="word-suggestion" onclick="manualAction('complete_word', {word: '${word}'})">${word}</button>`
            ).join('');
            
            this.elements.wordSuggestions.innerHTML = `
                <div class="small mb-2">Suggestions for "${currentWord}":</div>
                ${suggestionHtml}
            `;
        } else if (currentWord) {
            this.elements.wordSuggestions.innerHTML = 
                `<p class="text-muted small">No suggestions for "${currentWord}"</p>`;
        } else {
            this.elements.wordSuggestions.innerHTML = 
                '<p class="text-muted small">Type letters to see suggestions...</p>';
        }
    }
    
    getWordSuggestions(partial) {
        if (!partial) return [];
        
        const partialUpper = partial.toUpperCase();
        return this.vocabularyWords
            .filter(word => word.startsWith(partialUpper))
            .sort((a, b) => {
                // Prioritize shorter words and common words
                const scoreA = this.getWordScore(a);
                const scoreB = this.getWordScore(b);
                return scoreB - scoreA;
            })
            .slice(0, 5); // Limit to 5 suggestions
    }
    
    getWordScore(word) {
        // Simple scoring based on word frequency
        const frequencies = {
            'YES': 10, 'NO': 10,
            'HELP': 8, 'HI': 8,
            'STOP': 6, 'SO': 6, 'TO': 6,
            'IS': 4, 'IT': 4, 'OR': 4
        };
        return frequencies[word] || 1;
    }
    
    updateCharacterCounts() {
        const text = this.elements.currentText.textContent || '';
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        
        this.elements.wordCount.textContent = words;
        this.elements.charCount.textContent = text.length;
    }
    
    updateEventsDisplay(events) {
        const eventsHtml = events.slice(0, this.config.maxEvents).map(event => {
            const time = new Date(event.timestamp).toLocaleTimeString();
            const typeClass = `event-${event.type.toLowerCase().replace('_', '-')}`;
            
            let description = this.getEventDescription(event);
            
            return `
                <div class="event-item ${typeClass}">
                    <div class="d-flex justify-content-between">
                        <span>${description}</span>
                        <span class="event-timestamp">${time}</span>
                    </div>
                    ${event.confidence ? `<small>Confidence: ${(event.confidence * 100).toFixed(1)}%</small>` : ''}
                </div>
            `;
        }).join('');
        
        this.elements.recentEvents.innerHTML = eventsHtml || '<p class="text-muted small">No events yet...</p>';
    }
    
    getEventDescription(event) {
        switch(event.type) {
            case 'MOTOR_PREDICTION':
                const actions = ['Right Hand', 'Left Hand', 'Feet', 'Rest'];
                return `Motor: ${actions[event.predicted_class] || 'Unknown'}`;
            case 'P300_CONFIRMATION':
                return `P300: ${event.metadata?.predicted_word || 'Confirmation'}`;
            case 'LETTER_SELECTED':
                return `Selected: ${event.selected_letter}`;
            case 'WORD_COMPLETED':
                return `Completed: ${event.completed_word}`;
            case 'SIDE_SELECTED':
                return `Side: ${event.metadata?.selected_side}`;
            case 'SPACE_INSERTED':
                return 'Space inserted';
            case 'STATE_CHANGED':
                return `State: ${event.new_state}`;
            default:
                return event.type;
        }
    }
    
    animateLetterSelection(letter) {
        const btn = document.querySelector(`[data-letter="${letter}"]`);
        if (btn) {
            btn.classList.add('recently-selected');
            setTimeout(() => {
                btn.classList.remove('recently-selected');
            }, 1000);
        }
    }
    
    animateWordCompletion(word) {
        this.elements.currentWord.classList.add('success-flash');
        setTimeout(() => {
            this.elements.currentWord.classList.remove('success-flash');
        }, 600);
    }
    
    handleP300Confirmation(event) {
        // Visual feedback for P300 confirmation
        this.elements.p300Status.classList.add('success-flash');
        setTimeout(() => {
            this.elements.p300Status.classList.remove('success-flash');
        }, 600);
    }
    
    async manualAction(action, data = {}) {
        try {
            const formData = new FormData();
            formData.append('action', action);
            
            // Add additional data
            Object.keys(data).forEach(key => {
                formData.append(key, data[key]);
            });
            
            const response = await this.apiCall(`/action/${this.sessionId}/`, 'POST', formData);
            
            if (response.status === 'success') {
                // Trigger immediate status update
                this.fetchSessionStatus();
            } else {
                throw new Error(response.message);
            }
        } catch (error) {
            console.error('Manual action failed:', error);
            this.showNotification(`Action failed: ${error.message}`, 'error');
        }
    }
    
    async apiCall(endpoint, method = 'GET', body = null) {
        const url = `/communicator${endpoint}`;
        const options = {
            method: method,
            headers: {}
        };
        
        if (method === 'POST') {
            options.headers['X-CSRFToken'] = this.csrfToken;
            
            if (body instanceof FormData) {
                options.body = body;
            } else {
                options.headers['Content-Type'] = 'application/json';
                options.body = JSON.stringify(body);
            }
        }
        
        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
}

// Global functions for template usage
function manualAction(action, data = {}) {
    if (window.communicator) {
        window.communicator.manualAction(action, data);
    }
}

function showWordSuggestions() {
    const modal = new bootstrap.Modal(document.getElementById('wordSuggestionModal'));
    
    // Update modal content
    const currentWord = document.getElementById('currentWord').textContent;
    document.getElementById('modalCurrentWord').textContent = currentWord;
    
    const suggestions = window.communicator ? window.communicator.getWordSuggestions(currentWord) : [];
    const suggestionsDiv = document.getElementById('modalSuggestions');
    
    if (suggestions.length > 0) {
        suggestionsDiv.innerHTML = suggestions.map(word =>
            `<button class="btn btn-outline-primary me-2 mb-2" onclick="manualAction('complete_word', {word: '${word}'}); bootstrap.Modal.getInstance(document.getElementById('wordSuggestionModal')).hide();">${word}</button>`
        ).join('');
    } else {
        suggestionsDiv.innerHTML = '<p class="text-muted">No suggestions available</p>';
    }
    
    modal.show();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('BCI Communicator script loaded');
});