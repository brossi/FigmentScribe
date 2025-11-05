/**
 * Scribe Web UI - JavaScript
 *
 * Handles form submission, AJAX requests, and UI state management.
 */

(function() {
    'use strict';

    // State management
    let currentOutputFiles = {
        png: null,
        svg: null
    };

    /**
     * Show loading state
     */
    function showLoading() {
        $('#empty-state').hide();
        $('#output-state').hide();
        $('#error-state').hide();
        $('#loading-state').show();
        $('#generate-btn').prop('disabled', true).html('<span class="spinner-border spinner-border-sm me-2"></span> Generating...');
    }

    /**
     * Show error state
     */
    function showError(message) {
        $('#loading-state').hide();
        $('#empty-state').hide();
        $('#output-state').hide();
        $('#error-message').text(message);
        $('#error-state').show();
        $('#generate-btn').prop('disabled', false).html('<i class="bi bi-stars"></i> Generate Handwriting');
    }

    /**
     * Show output state
     */
    function showOutput(pngFilename, svgFilename, generationTime) {
        $('#loading-state').hide();
        $('#empty-state').hide();
        $('#error-state').hide();

        // Update image
        const pngUrl = `/download/${pngFilename}?t=${Date.now()}`;
        $('#output-image').attr('src', pngUrl);

        // Update generation time
        $('#generation-time').text(generationTime);

        // Update download links
        $('#download-png-btn').attr('href', `/download/${pngFilename}`).attr('download', pngFilename);
        $('#download-svg-btn').attr('href', `/download/${svgFilename}`).attr('download', svgFilename);

        // Store current files
        currentOutputFiles = {
            png: pngFilename,
            svg: svgFilename
        };

        // Show output
        $('#output-state').show();
        $('#generate-btn').prop('disabled', false).html('<i class="bi bi-stars"></i> Generate Handwriting');
    }

    /**
     * Reset to empty state
     */
    function showEmpty() {
        $('#loading-state').hide();
        $('#output-state').hide();
        $('#error-state').hide();
        $('#empty-state').show();
        $('#generate-btn').prop('disabled', false).html('<i class="bi bi-stars"></i> Generate Handwriting');
    }

    /**
     * Generate handwriting via AJAX
     */
    function generateHandwriting(formData) {
        showLoading();

        $.ajax({
            url: '/generate',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            timeout: 120000,  // 2 minute timeout
            success: function(response) {
                if (response.success) {
                    showOutput(
                        response.png_filename,
                        response.svg_filename,
                        response.generation_time
                    );
                } else {
                    showError(response.error || 'Generation failed');
                }
            },
            error: function(xhr, status, error) {
                let errorMessage = 'An error occurred';

                if (xhr.responseJSON && xhr.responseJSON.error) {
                    errorMessage = xhr.responseJSON.error;
                } else if (status === 'timeout') {
                    errorMessage = 'Request timed out (> 2 minutes)';
                } else if (xhr.status === 0) {
                    errorMessage = 'Cannot connect to server';
                } else {
                    errorMessage = `${xhr.status}: ${error}`;
                }

                showError(errorMessage);
            }
        });
    }

    /**
     * Handle form submission
     */
    $('#generate-form').on('submit', function(e) {
        e.preventDefault();

        // Get form data
        const text = $('#text-input').val().trim();
        const style = parseInt($('#style-select').val());
        const bias = parseFloat($('#bias-slider').val());
        const format = $('input[name="format"]:checked').val();

        // Validate
        if (!text) {
            showError('Please enter some text');
            return;
        }

        // Prepare request data
        const formData = {
            text: text,
            style: style,
            bias: bias,
            format: format
        };

        // Generate
        generateHandwriting(formData);
    });

    /**
     * Handle "Generate Again" button
     */
    $('#generate-again-btn').on('click', function() {
        showEmpty();
        $('#text-input').focus();
    });

    /**
     * Initialize app
     */
    $(document).ready(function() {
        console.log('Scribe Web UI initialized');

        // Check if model is loaded
        $.ajax({
            url: '/health',
            method: 'GET',
            success: function(response) {
                if (!response.model_loaded) {
                    showError('Model not loaded. Please check server logs.');
                }
            },
            error: function() {
                console.warn('Could not check health status');
            }
        });

        // Focus text input on load
        $('#text-input').focus();
    });

})();
