/**
 * BCI Communicator - Moving Window Frontend Implementation
 * This implements the exact moving window system with cooldown and prominent P300 word suggestions
 */

class BCIMovingWindowCommunicator {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.isRunning = false;
        this.pollingInterval = null;
        this.lastEventTimestamp = null;
        
        // Window positions (synced with backend)
        this.leftWindowIndex = 0;
        this.rightWindowIndex = 0;
        this.activeGroup = 'left';
        
        // Cooldown state
        this.inCooldown = false;
        this.cooldownDuration = 2000; // 2 seconds
        this.cooldownTimer = null;
        
        // P300 state
        this.p300Enabled = false;
        this.wordSuggestionsAvailable = false;
        
        // Configuration
        this.config = {
            pollingInterval: 150, // Fast polling for real-time response
            animationDuration: 800,
            cooldownDuration: 2000
        };
        
        // Motor imagery class mappings
        this.CLASSES = {
            RIGHT_HAND: 0,
            LEFT_HAND: 1,
            FEET: 2,
            REST: 3
        };
        
        this.initializeElements();
        this.setupEventListeners();
        this.updateWindowHighlights();
        
        console.log('🎮 BCI Moving Window Communicator initialized');
    }
    
    initializeElements() {
        this.elements = {
            // Control buttons
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            
            // Text display
            currentText: document.getElementById('currentText'),
            currentWord: document.getElementById('currentWord'),
            
            // Status displays
            systemStatus: document.getElementById('systemStatus'),
            cooldownStatus: document.getElementById('cooldownStatus'),
            windowPosition: document.getElementById('windowPosition'),
            lastPrediction: document.getElementById('lastPrediction'),
            
            // Letter groups
            leftLetters: document.querySelectorAll('#leftLetters .letter-btn'),
            rightLetters: document.querySelectorAll('#rightLetters .letter-btn'),
            
            // P300 word suggestions (prominent display)
            wordSuggestionsPanel: document.getElementById('wordSuggestionsPanel'),
            wordSuggestions: document.getElementById('wordSuggestions'),
            
            // Prediction bars
            motorImageryBars: document.getElementById('motorImageryPredictions'),
            
            // Cooldown indicator
            cooldownIndicator: document.getElementById('cooldownIndicator'),
            
            // Recent events
            recentEvents: document.getElementById('recentEvents')
        };
    }
    
    setupEventListeners() {
        // Control buttons
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener('click', () => this.startCommunication());
        }
        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', () => this.stopCommunication());
        }
        
        // Keyboard shortcuts for testing
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
    }
    
    handleKeyboardShortcuts(e) {
        if (e.ctrlKey) {
            switch(e.key) {
                case 's':
                    e.preventDefault();
                    this.isRunning ? this.stopCommunication() : this.startCommunication();
                    break;
                case '1':
                    e.preventDefault();
                    this.simulatePrediction(this.CLASSES.LEFT_HAND, 0.85);
                    break;
                case '2':
                    e.preventDefault();
                    this.simulatePrediction(this.CLASSES.RIGHT_HAND, 0.85);
                    break;
                case '3':
                    e.preventDefault();
                    this.simulatePrediction(this.CLASSES.REST, 0.85);
                    break;
                case ' ':
                    e.preventDefault();
                    this.simulatePrediction(this.CLASSES.FEET, 0.85);
                    break;
            }
        }
    }
    
    async startCommunication() {
        try {
            const response = await this.apiCall(`/api/start/${this.sessionId}/`, 'POST');
            
            if (response.status === 'success') {
                this.isRunning = true;
                this.updateControlButtons();
                this.startPolling();
                
                this.showNotification('🚀 BCI Moving Window System Started!', 'success');
                console.log('✅ BCI system started');
            } else {
                throw new Error(response.message);
            }
        } catch (error) {
            console.error('❌ Failed to start communication:', error);
            this.showNotification(`Failed to start: ${error.message}`, 'error');
        }
    }
    
    async stopCommunication() {
        try {
            const response = await this.apiCall(`/api/stop/${this.sessionId}/`, 'POST');
            
            this.isRunning = false;
            this.updateControlButtons();
            this.stopPolling();
            
            this.showNotification('🛑 Communication system stopped', 'info');
            console.log('🛑 BCI system stopped');
        } catch (error) {
            console.error('❌ Failed to stop communication:', error);
            this.showNotification(`Failed to stop: ${error.message}`, 'error');
        }
    }
    
    updateControlButtons() {
        if (this.elements.startBtn && this.elements.stopBtn) {
            this.elements.startBtn.disabled = this.isRunning;
            this.elements.stopBtn.disabled = !this.isRunning;
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
            const response = await this.apiCall(`/api/status/${this.sessionId}/`);
            this.updateInterface(response);
        } catch (error) {
            console.error('❌ Failed to fetch session status:', error);
        }
    }
    
    updateInterface(data) {
        const session = data.session;
        const events = data.events;
        const systemStatus = data.system_status;
        
        // Update session display
        this.updateSessionDisplay(session);
        
        // Process new events for window movements
        this.processEventsForMovingWindows(events);
        
        // Update system status
        if (systemStatus && systemStatus.state_info) {
            this.updateWindowPositions(systemStatus.state_info);
            this.updateCooldownStatus(systemStatus.state_info);
            this.updateP300Status(systemStatus.state_info);
        }
        
        // Update word suggestions
        this.updateWordSuggestions(session.current_word);
    }
    
    processEventsForMovingWindows(events) {
        const newEvents = this.getNewEvents(events);
        
        newEvents.forEach(event => {
            console.log(`📡 Processing event: ${event.type}`, event);
            
            switch(event.type) {
                case 'MOTOR_PREDICTION':
                    this.handleMotorPredictionEvent(event);
                    break;
                case 'LETTER_SELECTED':
                    this.handleLetterSelectionEvent(event);
                    break;
                case 'WORD_COMPLETED':
                    this.handleWordCompletionEvent(event);
                    break;
                case 'SPACE_INSERTED':
                    this.handleSpaceInsertionEvent(event);
                    break;
                case 'P300_CONFIRMATION':
                    this.handleP300ConfirmationEvent(event);
                    break;
                case 'STATE_CHANGED':
                    if (event.new_state === 'WORD_SUGGESTIONS_AVAILABLE') {
                        this.handleWordSuggestionsAvailable(event);
                    }
                    break;
            }
        });
        
        // Update recent events display
        this.updateRecentEventsDisplay(events);
    }
    
    handleMotorPredictionEvent(event) {
        const { predicted_class, confidence, metadata } = event;
        
        // Update prediction bars
        this.updateMotorImageryBars(predicted_class, confidence);
        
        if (metadata) {
            const { action, active_group, left_window_index, right_window_index } = metadata;
            
            // Update local window positions
            if (left_window_index !== undefined) this.leftWindowIndex = left_window_index;
            if (right_window_index !== undefined) this.rightWindowIndex = right_window_index;
            if (active_group) this.activeGroup = active_group;
            
            // Animate window movement
            switch(action) {
                case 'move_window_left':
                    this.animateWindowMovement('left', active_group);
                    break;
                case 'move_window_right':
                    this.animateWindowMovement('right', active_group);
                    break;
            }
            
            // Update window highlights
            this.updateWindowHighlights();
            
            // Start cooldown visualization
            this.startCooldownVisualization();
        }
        
        // Update last prediction display
        const classNames = ['Right Hand', 'Left Hand', 'Feet', 'Rest'];
        if (this.elements.lastPrediction) {
            this.elements.lastPrediction.textContent = `${classNames[predicted_class]} (${(confidence * 100).toFixed(1)}%)`;
        }
    }
    
    handleLetterSelectionEvent(event) {
        const { selected_letter, metadata } = event;
        
        console.log(`✅ Letter selected: ${selected_letter}`);
        
        // Animate letter confirmation
        this.animateLetterConfirmation(selected_letter, metadata?.window_source);
        
        // Show feedback
        this.showActionFeedback(`Letter selected: ${selected_letter}`, 'success');
    }
    
    handleWordCompletionEvent(event) {
        const { completed_word } = event;
        
        console.log(`🎯 Word completed: ${completed_word}`);
        
        // Animate word completion
        this.animateWordCompletion(completed_word);
        
        // Show feedback
        this.showActionFeedback(`Word completed: ${completed_word}`, 'success');
    }
    
    handleSpaceInsertionEvent(event) {
        console.log('⎵ Space inserted');
        this.showActionFeedback('Space inserted', 'info');
    }
    
    handleP300ConfirmationEvent(event) {
        const { confidence, metadata } = event;
        const predicted_word = metadata?.predicted_word;
        
        console.log(`👁️ P300 confirmation: ${predicted_word} (${(confidence * 100).toFixed(1)}%)`);
        
        if (predicted_word) {
            this.animateP300Selection(predicted_word);
        }
    }
    
    handleWordSuggestionsAvailable(event) {
        const { metadata } = event;
        
        if (metadata && metadata.suggestions) {
            this.displayWordSuggestions(metadata.suggestions);
            this.p300Enabled = metadata.p300_enabled || false;
        }
    }
    
    // MOVING WINDOW VISUALIZATION
    updateWindowHighlights() {
        // Clear all existing highlights
        document.querySelectorAll('.letter-btn').forEach(btn => {
            btn.classList.remove('window-highlight', 'window-active');
        });
        
        // Highlight left window
        if (this.elements.leftLetters[this.leftWindowIndex]) {
            this.elements.leftLetters[this.leftWindowIndex].classList.add('window-highlight');
            if (this.activeGroup === 'left') {
                this.elements.leftLetters[this.leftWindowIndex].classList.add('window-active');
            }
        }
        
        // Highlight right window
        if (this.elements.rightLetters[this.rightWindowIndex]) {
            this.elements.rightLetters[this.rightWindowIndex].classList.add('window-highlight');
            if (this.activeGroup === 'right') {
                this.elements.rightLetters[this.rightWindowIndex].classList.add('window-active');
            }
        }
        
        // Update window position display
        if (this.elements.windowPosition) {
            const leftLetter = this.elements.leftLetters[this.leftWindowIndex]?.textContent || '?';
            const rightLetter = this.elements.rightLetters[this.rightWindowIndex]?.textContent || '?';
            this.elements.windowPosition.textContent = `L:${leftLetter} R:${rightLetter}`;
        }
    }
    
    animateWindowMovement(direction, group) {
        const targetGroup = group === 'left' ? this.elements.leftLetters : this.elements.rightLetters;
        const targetIndex = group === 'left' ? this.leftWindowIndex : this.rightWindowIndex;
        
        // Add movement animation class
        if (targetGroup[targetIndex]) {
            targetGroup[targetIndex].classList.add('window-moving');
            setTimeout(() => {
                targetGroup[targetIndex].classList.remove('window-moving');
            }, this.config.animationDuration);
        }
        
        console.log(`🎯 Window moved ${direction} in ${group} group`);
    }
    
    animateLetterConfirmation(letter, windowSource) {
        const btn = document.querySelector(`[data-letter="${letter}"]`);
        if (btn) {
            btn.classList.add('letter-confirmed');
            setTimeout(() => {
                btn.classList.remove('letter-confirmed');
            }, this.config.animationDuration);
        }
    }
    
    animateWordCompletion(word) {
        if (this.elements.currentWord) {
            this.elements.currentWord.classList.add('word-completed');
            setTimeout(() => {
                this.elements.currentWord.classList.remove('word-completed');
            }, this.config.animationDuration);
        }
    }
    
    // COOLDOWN VISUALIZATION
    startCooldownVisualization() {
        this.inCooldown = true;
        
        // Show cooldown indicator
        if (this.elements.cooldownIndicator) {
            this.elements.cooldownIndicator.classList.add('active');
        }
        
        // Update cooldown status
        if (this.elements.cooldownStatus) {
            this.elements.cooldownStatus.textContent = 'Active';
            this.elements.cooldownStatus.className = 'status-value cooldown-active';
        }
        
        // Start countdown
        let remaining = this.config.cooldownDuration;
        const countdownInterval = setInterval(() => {
            remaining -= 100;
            
            if (this.elements.cooldownStatus) {
                this.elements.cooldownStatus.textContent = `${(remaining / 1000).toFixed(1)}s`;
            }
            
            if (remaining <= 0) {
                clearInterval(countdownInterval);
                this.endCooldownVisualization();
            }
        }, 100);
    }
    
    endCooldownVisualization() {
        this.inCooldown = false;
        
        // Hide cooldown indicator
        if (this.elements.cooldownIndicator) {
            this.elements.cooldownIndicator.classList.remove('active');
        }
        
        // Update cooldown status
        if (this.elements.cooldownStatus) {
            this.elements.cooldownStatus.textContent = 'Ready';
            this.elements.cooldownStatus.className = 'status-value';
        }
        
        console.log('✅ Cooldown visualization ended');
    }
    
    // PROMINENT P300 WORD SUGGESTIONS
    displayWordSuggestions(suggestions) {
        if (!this.elements.wordSuggestions) return;
        
        if (suggestions && suggestions.length > 0) {
            const suggestionsHtml = suggestions.map(word => `
                <div class="word-suggestion-item" 
                     data-word="${word}" 
                     onclick="window.bciCommunicator.selectWordSuggestion('${word}')">
                    ${word}
                </div>
            `).join('');
            
            this.elements.wordSuggestions.innerHTML = suggestionsHtml;
            
            // Show P300 panel prominently
            if (this.elements.wordSuggestionsPanel) {
                this.elements.wordSuggestionsPanel.classList.add('p300-active');
            }
            
            console.log(`💡 Displaying ${suggestions.length} word suggestions for P300`);
        } else {
            this.elements.wordSuggestions.innerHTML = '<div class="text-muted">Type letters to see word suggestions...</div>';
            
            if (this.elements.wordSuggestionsPanel) {
                this.elements.wordSuggestionsPanel.classList.remove('p300-active');
            }
        }
    }
    
    selectWordSuggestion(word) {
        console.log(`👁️ Word suggestion clicked: ${word} (simulating P300)`);
        
        // Animate P300 selection
        this.animateP300Selection(word);
        
        // In real implementation, this would be detected by P300 model
        // For now, we simulate the selection
        setTimeout(() => {
            this.simulateWordCompletion(word);
        }, 1000);
    }
    
    animateP300Selection(word) {
        const wordBtn = document.querySelector(`[data-word="${word}"]`);
        if (wordBtn) {
            wordBtn.classList.add('p300-focused');
            setTimeout(() => {
                wordBtn.classList.remove('p300-focused');
            }, 1000);
        }
    }
    
    // UPDATE METHODS
    updateSessionDisplay(session) {
        if (this.elements.currentText) {
            this.elements.currentText.textContent = session.current_text || '';
        }
        if (this.elements.currentWord) {
            this.elements.currentWord.textContent = session.current_word || '';
        }
    }
    
    updateWindowPositions(stateInfo) {
        if (stateInfo.left_window_index !== undefined) {
            this.leftWindowIndex = stateInfo.left_window_index;
        }
        if (stateInfo.right_window_index !== undefined) {
            this.rightWindowIndex = stateInfo.right_window_index;
        }
        if (stateInfo.active_group) {
            this.activeGroup = stateInfo.active_group;
        }
        
        this.updateWindowHighlights();
    }
    
    updateCooldownStatus(stateInfo) {
        if (stateInfo.in_cooldown !== undefined) {
            this.inCooldown = stateInfo.in_cooldown;
            
            if (stateInfo.cooldown_remaining && stateInfo.cooldown_remaining > 0) {
                if (this.elements.cooldownStatus) {
                    this.elements.cooldownStatus.textContent = `${stateInfo.cooldown_remaining.toFixed(1)}s`;
                    this.elements.cooldownStatus.className = 'status-value cooldown-active';
                }
            } else if (!this.inCooldown) {
                if (this.elements.cooldownStatus) {
                    this.elements.cooldownStatus.textContent = 'Ready';
                    this.elements.cooldownStatus.className = 'status-value';
                }
            }
        }
    }
    
    updateP300Status(stateInfo) {
        if (stateInfo.p300_enabled !== undefined) {
            this.p300Enabled = stateInfo.p300_enabled;
            // Update P300 status display if element exists
        }
    }
    
    updateMotorImageryBars(predictedClass, confidence) {
        if (!this.elements.motorImageryBars) return;
        
        // Reset all bars
        const bars = this.elements.motorImageryBars.querySelectorAll('.confidence-fill');
        bars.forEach(bar => bar.style.width = '0%');
        
        // Set the predicted class bar
        if (bars[predictedClass]) {
            bars[predictedClass].style.width = (confidence * 100) + '%';
        }
    }
    
    updateWordSuggestions(currentWord) {
        // This will be handled by the STATE_CHANGED event from backend
        // when word suggestions become available
    }
    
    updateRecentEventsDisplay(events) {
        if (!this.elements.recentEvents) return;
        
        const eventsHtml = events.slice(0, 5).map(event => {
            const timeStr = new Date(event.timestamp).toLocaleTimeString();
            return `
                <div class="event-item">
                    <div class="d-flex justify-content-between">
                        <span>${this.getEventDescription(event)}</span>
                        <small class="text-muted">${timeStr}</small>
                    </div>
                </div>
            `;
        }).join('');
        
        this.elements.recentEvents.innerHTML = eventsHtml || '<p class="text-muted small">No events yet...</p>';
    }
    
    getEventDescription(event) {
        switch(event.type) {
            case 'MOTOR_PREDICTION':
                const actions = ['Right Hand', 'Left Hand', 'Feet', 'Rest'];
                return `🧠 ${actions[event.predicted_class] || 'Unknown'}`;
            case 'LETTER_SELECTED':
                return `✅ Selected: ${event.selected_letter}`;
            case 'WORD_COMPLETED':
                return `🎯 Completed: ${event.completed_word}`;
            case 'SPACE_INSERTED':
                return `⎵ Space inserted`;
            case 'P300_CONFIRMATION':
                return `👁️ P300: ${event.metadata?.predicted_word || 'Confirmation'}`;
            default:
                return event.type;
        }
    }
    
    // UTILITY METHODS
    getNewEvents(events) {
        if (!events || events.length === 0) {
            return [];
        }
        
        if (!this.lastEventTimestamp) {
            this.lastEventTimestamp = events[0].timestamp;
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
    
    async apiCall(url, method = 'GET', data = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.getCsrfToken()
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(`/communicator${url}`, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    getCsrfToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';
    }
    
    showNotification(message, type) {
        console.log(`${type.toUpperCase()}: ${message}`);
        
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `alert alert-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'} position-fixed`;
        toast.style.top = '20px';
        toast.style.right = '20px';
        toast.style.zIndex = '9999';
        toast.innerHTML = `<i class="fas fa-robot"></i> ${message}`;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 3000);
    }
    
    showActionFeedback(message, type) {
        this.showNotification(message, type);
    }
    
    // TESTING/SIMULATION METHODS
    simulatePrediction(classIndex, confidence) {
        if (!this.isRunning || this.inCooldown) {
            console.log('🚫 Simulation ignored - system not running or in cooldown');
            return;
        }
        
        const classNames = ['Right Hand', 'Left Hand', 'Feet', 'Rest'];
        console.log(`🧪 Simulating prediction: ${classNames[classIndex]} (${(confidence * 100).toFixed(1)}%)`);
        
        // This would normally come from the backend via polling
        // For testing, we can simulate the prediction process
        this.handleMotorPredictionEvent({
            type: 'MOTOR_PREDICTION',
            predicted_class: classIndex,
            confidence: confidence,
            metadata: {
                action: classIndex === 1 ? 'move_window_left' : classIndex === 0 ? 'move_window_right' : 'other',
                active_group: this.activeGroup,
                left_window_index: this.leftWindowIndex,
                right_window_index: this.rightWindowIndex
            }
        });
    }
    
    simulateWordCompletion(word) {
        // Simulate word completion
        this.handleWordCompletionEvent({
            type: 'WORD_COMPLETED',
            completed_word: word
        });
    }
}

// Global initialization
document.addEventListener('DOMContentLoaded', function() {
    const sessionId = window.sessionId; // Set in template
    if (sessionId) {
        window.bciCommunicator = new BCIMovingWindowCommunicator(sessionId);
        console.log('🎮 BCI Moving Window Communicator initialized');
        
        // Add help text
        console.log('🎮 Keyboard shortcuts for testing:');
        console.log('   Ctrl+1: Simulate LEFT_HAND prediction');
        console.log('   Ctrl+2: Simulate RIGHT_HAND prediction');
        console.log('   Ctrl+3: Simulate REST prediction');
        console.log('   Ctrl+Space: Simulate FEET prediction');
        console.log('   Ctrl+S: Start/Stop system');
    } else {
        console.error('❌ Session ID not found - cannot initialize BCI Communicator');
    }
});

// CSS for moving window animations
const style = document.createElement('style');
style.textContent = `
    .window-highlight {
        background: linear-gradient(45deg, #ffd700, #ffed4a) !important;
        color: #333 !important;
        border-color: #ffd700 !important;
        transform: scale(1.15) !important;
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6) !important;
        z-index: 10 !important;
        animation: window-glow 2s ease-in-out infinite !important;
    }
    
    .window-active {
        animation: window-pulse 1s ease-in-out infinite !important;
    }
    
    .window-moving {
        animation: window-move 0.5s ease-out !important;
    }
    
    .letter-confirmed {
        background: linear-gradient(45deg, #28a745, #20c997) !important;
        animation: confirm-pulse 1s ease-out !important;
    }
    
    .word-completed {
        animation: word-success 1s ease-out !important;
    }
    
    .p300-focused {
        animation: p300-focus 1s ease-in-out !important;
        background: linear-gradient(45deg, #ff6b6b, #ee5a52) !important;
        transform: scale(1.2) !important;
    }
    
    .p300-active {
        border: 3px solid #ffd700 !important;
        box-shadow: 0 0 20px rgba(255, 215, 0, 0.5) !important;
    }
    
    .cooldown-active {
        color: #ffc107 !important;
        font-weight: bold !important;
    }
    
    @keyframes window-glow {
        0%, 100% { box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6); }
        50% { box-shadow: 0 12px 35px rgba(255, 215, 0, 0.9); }
    }
    
    @keyframes window-pulse {
        0%, 100% { transform: scale(1.15); }
        50% { transform: scale(1.25); }
    }
    
    @keyframes window-move {
        0% { transform: scale(1.15) rotate(-5deg); }
        50% { transform: scale(1.3) rotate(5deg); }
        100% { transform: scale(1.15) rotate(0deg); }
    }
    
    @keyframes confirm-pulse {
        0% { transform: scale(1.15); }
        50% { transform: scale(1.4); }
        100% { transform: scale(1); }
    }
    
    @keyframes word-success {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); background-color: #28a745; }
        100% { transform: scale(1); }
    }
    
    @keyframes p300-focus {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.8; }
        100% { transform: scale(1.2); opacity: 1; }
    }
`;
document.head.appendChild(style);