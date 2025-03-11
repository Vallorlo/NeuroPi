document.addEventListener('DOMContentLoaded', function() {
    const startCaptureBtn = document.getElementById('startCaptureBtn');
    const timestampBtn = document.getElementById('timestampBtn');
    const captureForm = document.getElementById('captureForm');
    const timestampsContainer = document.getElementById('timestampsContainer');
    const timestampsList = document.getElementById('timestampsList');
    
    let captureStartTime = null;
    let isCapturing = false;

    // Check if there are existing timestamps
    if (timestampsList && timestampsList.children.length > 0) {
        timestampsContainer.style.display = 'block';
    }

    // Start capture button
    startCaptureBtn.addEventListener('click', function(e) {
        e.preventDefault();
        
        // Submit form to start capture
        isCapturing = true;
        captureStartTime = new Date();
        
        // Show timestamp button
        timestampBtn.style.display = 'inline-block';
        timestampBtn.disabled = false;
        
        // Disable start button
        startCaptureBtn.disabled = true;
        
        // Show timestamps container
        timestampsContainer.style.display = 'block';
        
        // Clear existing timestamps from the current attempt
        while (timestampsList.firstChild) {
            timestampsList.removeChild(timestampsList.firstChild);
        }
        
        // Submit the form to start capture
        captureForm.submit();
    });

    // Timestamp button
    timestampBtn.addEventListener('click', function() {
        const currentTime = new Date();
        const elapsedMs = currentTime - captureStartTime;
        
        // Format elapsed time as minutes:seconds.milliseconds
        const minutes = Math.floor(elapsedMs / 60000);
        const seconds = Math.floor((elapsedMs % 60000) / 1000);
        const ms = elapsedMs % 1000;
        const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
        
        // Add to the list
        const listItem = document.createElement('li');
        listItem.className = 'list-group-item';
        listItem.textContent = `Time: ${formattedTime}`;
        timestampsList.appendChild(listItem);
        
        // Send to server via AJAX
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        
        fetch(window.location.href, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'X-CSRFToken': csrfToken
            },
            body: new URLSearchParams({
                'timestamp_event': 'true',
                'timestamp': formattedTime
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('Timestamp saved:', data);
        })
        .catch(error => {
            console.error('Error saving timestamp:', error);
        });
    });
});