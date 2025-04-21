document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize chart displays for Plotly charts
    const driverChartElement = document.getElementById('driver-performance-chart');
    if (driverChartElement && driverChartElement.dataset.chart) {
        const driverData = JSON.parse(driverChartElement.dataset.chart);
        Plotly.newPlot('driver-performance-chart', driverData.data, driverData.layout);
    }

    const issueChartElement = document.getElementById('issue-distribution-chart');
    if (issueChartElement && issueChartElement.dataset.chart) {
        const issueData = JSON.parse(issueChartElement.dataset.chart);
        Plotly.newPlot('issue-distribution-chart', issueData.data, issueData.layout);
    }

    // Camera access for customer feedback
    const cameraButton = document.getElementById('camera-button');
    const cameraPreview = document.getElementById('camera-preview');
    const cameraContainer = document.getElementById('camera-container');
    const submitImageButton = document.getElementById('submit-image');
    const cancelImageButton = document.getElementById('cancel-image');
    
    let stream = null;

    if (cameraButton) {
        cameraButton.addEventListener('click', async function() {
            try {
                // Ask for camera permission
                const result = confirm('Allow access to your camera to upload an image of the damaged product?');
                
                if (result) {
                    stream = await navigator.mediaDevices.getUserMedia({ video: true });
                    cameraPreview.srcObject = stream;
                    cameraContainer.classList.remove('d-none');
                    cameraButton.classList.add('d-none');
                }
            } catch (err) {
                console.error('Error accessing camera:', err);
                alert('Could not access the camera. Please check your permissions.');
            }
        });
    }

    if (submitImageButton) {
        submitImageButton.addEventListener('click', function() {
            if (stream) {
                // In a real app, we would capture and upload the image
                // For this demo, we'll just show a success message
                alert('Image submitted successfully!');
                
                // Clean up
                stream.getTracks().forEach(track => track.stop());
                cameraPreview.srcObject = null;
                cameraContainer.classList.add('d-none');
                cameraButton.classList.remove('d-none');
            }
        });
    }

    if (cancelImageButton) {
        cancelImageButton.addEventListener('click', function() {
            if (stream) {
                // Clean up
                stream.getTracks().forEach(track => track.stop());
                cameraPreview.srcObject = null;
                cameraContainer.classList.add('d-none');
                cameraButton.classList.remove('d-none');
            }
        });
    }

    // Packaging prediction form
    const packagingForm = document.getElementById('packaging-form');
    const weightInput = document.getElementById('weight_kg');
    
    if (packagingForm && weightInput) {
        weightInput.addEventListener('input', function() {
            // Validate weight input
            const value = parseFloat(this.value);
            if (isNaN(value) || value <= 0) {
                this.setCustomValidity('Please enter a valid weight greater than 0');
            } else if (value > 100) {
                this.setCustomValidity('Weight should be less than 100 kg');
            } else {
                this.setCustomValidity('');
            }
        });
    }

    // Driver report form
    const driverReportForm = document.getElementById('driver-report-form');
    const weatherSelect = document.getElementById('weather');
    const roadConditionSelect = document.getElementById('road_condition');
    const causeOfIssueSelect = document.getElementById('cause_of_issue');
    
    if (driverReportForm && weatherSelect && roadConditionSelect && causeOfIssueSelect) {
        // Update cause of issue based on weather and road condition
        function updateCauseOfIssue() {
            const weather = weatherSelect.value;
            const roadCondition = roadConditionSelect.value;
            
            // Simple rule-based suggestions
            if (weather === 'Rainy' || weather === 'Foggy') {
                if (roadCondition === 'Potholes' || roadCondition === 'Uneven') {
                    causeOfIssueSelect.value = 'Road_Shock';
                }
            } else if (weather === 'Dusty' && (roadCondition === 'Cracks' || roadCondition === 'Bumpy')) {
                causeOfIssueSelect.value = 'Heat_Damage';
            } else if (roadCondition === 'Smooth' || roadCondition === 'Good') {
                causeOfIssueSelect.value = 'None';
            }
        }
        
        weatherSelect.addEventListener('change', updateCauseOfIssue);
        roadConditionSelect.addEventListener('change', updateCauseOfIssue);
    }
}); 