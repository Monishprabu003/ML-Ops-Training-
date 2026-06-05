/**
 * PlacementMentorAI - Interview Behaviour Analysis
 * Frontend JavaScript for image/video upload and analysis
 */

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────
const MAX_IMAGE_SIZE = 16 * 1024 * 1024;  // 16 MB
const MAX_VIDEO_SIZE = 100 * 1024 * 1024; // 100 MB

// ─────────────────────────────────────────────
// DOM Elements
// ─────────────────────────────────────────────
const imageUploadArea = document.getElementById('imageUploadArea');
const imageInput = document.getElementById('imageInput');
const imageSpinner = document.getElementById('imageSpinner');
const imageLoadingOverlay = document.getElementById('imageLoadingOverlay');

const videoUploadArea = document.getElementById('videoUploadArea');
const videoInput = document.getElementById('videoInput');
const videoSpinner = document.getElementById('videoSpinner');
const videoLoadingOverlay = document.getElementById('videoLoadingOverlay');

// ─────────────────────────────────────────────
// Image Upload Handlers
// ─────────────────────────────────────────────
if (imageUploadArea) {
    imageUploadArea.addEventListener('click', () => imageInput.click());

    imageUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        imageUploadArea.classList.add('dragover');
    });

    imageUploadArea.addEventListener('dragleave', () => {
        imageUploadArea.classList.remove('dragover');
    });

    imageUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        imageUploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            imageInput.files = e.dataTransfer.files;
            analyzeImage();
        }
    });

    imageInput.addEventListener('change', analyzeImage);
}

// ─────────────────────────────────────────────
// Video Upload Handlers
// ─────────────────────────────────────────────
if (videoUploadArea) {
    videoUploadArea.addEventListener('click', () => videoInput.click());

    videoUploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoUploadArea.classList.add('dragover');
    });

    videoUploadArea.addEventListener('dragleave', () => {
        videoUploadArea.classList.remove('dragover');
    });

    videoUploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        videoUploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            videoInput.files = e.dataTransfer.files;
            analyzeVideo();
        }
    });

    videoInput.addEventListener('change', analyzeVideo);
}

// ─────────────────────────────────────────────
// Image Analysis
// ─────────────────────────────────────────────
async function analyzeImage() {
    if (!imageInput.files.length) return;

    const file = imageInput.files[0];

    // Validate file size
    if (file.size > MAX_IMAGE_SIZE) {
        showError('imageError', `File too large. Maximum size: ${formatBytes(MAX_IMAGE_SIZE)}`);
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    showLoading('image');
    hideError('imageError');
    document.getElementById('imageNoResults').style.display = 'none';
    document.getElementById('imageResults').style.display = 'none';

    try {
        const response = await fetch('/analyze-image', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            showError('imageError', result.error);
        } else {
            displayImageResults(result);
        }
    } catch (error) {
        showError('imageError', 'Network error: ' + error.message);
    } finally {
        hideLoading('image');
    }
}

function displayImageResults(result) {
    // Behaviour badge
    const behaviourClass = {
        'Confident': 'behaviour-confident',
        'Nervous': 'behaviour-nervous',
        'Distracted': 'behaviour-distracted'
    }[result.behaviour] || 'behaviour-confident';

    document.getElementById('imageBehaviourBadge').textContent = result.behaviour;
    document.getElementById('imageBehaviourBadge').className = `behaviour-badge ${behaviourClass}`;

    // Confidence score
    const confidence = Math.round(result.confidence_score * 100);
    document.getElementById('imageConfidenceScore').textContent = confidence + '%';
    document.getElementById('imageConfidenceBar').style.width = confidence + '%';

    // Behaviour score
    const behaviourScore = result.behaviour_score;
    document.getElementById('imageBehaviourScore').textContent = behaviourScore.toFixed(1) + '/100';
    document.getElementById('imageBehaviourScoreBar').style.width = behaviourScore + '%';

    // Features
    const features = result.features;
    let featuresHTML = '';
    if (features) {
        const featureItems = [
            { label: 'Head Tilt Angle', value: features.head_tilt_angle.toFixed(1) + '°', icon: '🔄' },
            { label: 'Spine Alignment', value: features.spine_alignment_score.toFixed(1) + '%', icon: '🦴' },
            { label: 'Hand Movement', value: features.hand_movement_delta.toFixed(1) + '%', icon: '✋' },
            { label: 'Eye Contact', value: features.eye_contact_proxy.toFixed(1) + '%', icon: '👁️' },
            { label: 'Shoulder Tension', value: features.shoulder_tension_ratio.toFixed(1) + '%', icon: '💪' },
        ];

        featureItems.forEach(item => {
            featuresHTML += `<div class="feature-card">
                <div class="d-flex justify-content-between align-items-center">
                    <span>${item.icon} ${item.label}</span>
                    <strong>${item.value}</strong>
                </div>
            </div>`;
        });
    }
    document.getElementById('imageFeatures').innerHTML = featuresHTML;

    // Dominant signal
    document.getElementById('imageDominantSignal').textContent =
        result.dominant_signal.replace(/_/g, ' ').toUpperCase();

    // Summary
    document.getElementById('imageSummary').textContent = result.summary;

    // Processed image
    if (result.processed_image_path) {
        document.getElementById('imagePreviewed').src = result.processed_image_path;
        document.getElementById('imagePreviewed').style.display = 'block';
    }

    // Show results
    document.getElementById('imageResults').style.display = 'block';
    document.getElementById('imageNoResults').style.display = 'none';

    // Update combined score if placement_score exists
    updateCombinedScore(behaviourScore);
}

// ─────────────────────────────────────────────
// Video Analysis
// ─────────────────────────────────────────────
async function analyzeVideo() {
    if (!videoInput.files.length) return;

    const file = videoInput.files[0];

    // Validate file size
    if (file.size > MAX_VIDEO_SIZE) {
        showError('videoError', `File too large. Maximum size: ${formatBytes(MAX_VIDEO_SIZE)}`);
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    showLoading('video');
    hideError('videoError');
    document.getElementById('videoNoResults').style.display = 'none';
    document.getElementById('videoResults').style.display = 'none';

    try {
        const response = await fetch('/analyze-video', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.error) {
            showError('videoError', result.error);
        } else {
            displayVideoResults(result);
        }
    } catch (error) {
        showError('videoError', 'Network error: ' + error.message);
    } finally {
        hideLoading('video');
    }
}

function displayVideoResults(result) {
    // Behaviour badge
    const behaviourClass = {
        'Confident': 'behaviour-confident',
        'Nervous': 'behaviour-nervous',
        'Distracted': 'behaviour-distracted'
    }[result.behaviour] || 'behaviour-confident';

    document.getElementById('videoBehaviourBadge').textContent = result.behaviour;
    document.getElementById('videoBehaviourBadge').className = `behaviour-badge ${behaviourClass}`;

    // Confidence score
    const confidence = Math.round(result.confidence_score * 100);
    document.getElementById('videoConfidenceScore').textContent = confidence + '%';
    document.getElementById('videoConfidenceBar').style.width = confidence + '%';

    // Behaviour score
    const behaviourScore = result.behaviour_score;
    document.getElementById('videoBehaviourScore').textContent = behaviourScore.toFixed(1) + '/100';
    document.getElementById('videoBehaviourScoreBar').style.width = behaviourScore + '%';

    // Behaviour distribution
    const distribution = result.behaviour_distribution;
    let distributionHTML = '';
    if (distribution) {
        const items = [
            { label: 'Confident', value: Math.round(distribution.confident * 100), color: 'bg-success' },
            { label: 'Nervous', value: Math.round(distribution.nervous * 100), color: 'bg-warning' },
            { label: 'Distracted', value: Math.round(distribution.distracted * 100), color: 'bg-danger' },
        ];

        items.forEach(item => {
            distributionHTML += `<div class="feature-card">
                <div class="d-flex justify-content-between mb-2">
                    <span>${item.label}</span>
                    <strong>${item.value}%</strong>
                </div>
                <div class="progress" style="height: 15px;">
                    <div class="progress-bar ${item.color}" style="width: ${item.value}%"></div>
                </div>
            </div>`;
        });
    }
    document.getElementById('videoBehaviourDistribution').innerHTML = distributionHTML;

    // Analysis details
    document.getElementById('videoAnalysisDetails').textContent =
        `Total frames analyzed: ${result.total_frames_analyzed}`;

    // Dominant signal
    document.getElementById('videoDominantSignal').textContent =
        result.dominant_signal.replace(/_/g, ' ').toUpperCase();

    // Summary
    document.getElementById('videoSummary').textContent = result.summary;

    // Processed video
    if (result.processed_video_path) {
        const videoEl = document.getElementById('videoPreviewed');
        videoEl.src = result.processed_video_path;
        videoEl.style.display = 'block';
        videoEl.load();
    }

    // Show results
    document.getElementById('videoResults').style.display = 'block';
    document.getElementById('videoNoResults').style.display = 'none';

    // Update combined score
    updateCombinedScore(behaviourScore);
}

// ─────────────────────────────────────────────
// Combined Score
// ─────────────────────────────────────────────
function updateCombinedScore(behaviourScore) {
    const urlParams = new URLSearchParams(window.location.search);
    const placementScore = parseFloat(urlParams.get('placement_score'));

    if (!isNaN(placementScore) && placementScore > 0) {
        const finalScore = (placementScore * 0.7) + (behaviourScore * 0.3);

        document.getElementById('placementScoreDisplay').textContent = placementScore.toFixed(1);
        document.getElementById('behaviourScoreDisplay').textContent = behaviourScore.toFixed(1);
        document.getElementById('finalScoreDisplay').textContent = finalScore.toFixed(1);
        document.getElementById('finalScoreBar').style.width = finalScore + '%';
        document.getElementById('finalScorePercentage').textContent = finalScore.toFixed(1) + '%';

        // Color-code the final score bar
        const bar = document.getElementById('finalScoreBar');
        if (finalScore >= 80) {
            bar.className = 'progress-bar progress-bar-custom bg-success';
        } else if (finalScore >= 60) {
            bar.className = 'progress-bar progress-bar-custom bg-warning';
        } else {
            bar.className = 'progress-bar progress-bar-custom bg-danger';
        }

        document.getElementById('combinedScoreCard').style.display = 'block';
    }
}

// ─────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────
function showLoading(type) {
    const overlay = document.getElementById(type + 'LoadingOverlay');
    if (overlay) overlay.style.display = 'flex';
}

function hideLoading(type) {
    const overlay = document.getElementById(type + 'LoadingOverlay');
    if (overlay) overlay.style.display = 'none';
}

function showError(elementId, message) {
    let errorEl = document.getElementById(elementId);
    if (!errorEl) {
        // Create error element dynamically
        errorEl = document.createElement('div');
        errorEl.id = elementId;
        errorEl.className = 'alert alert-danger alert-dismissible fade show mt-3';
        const parent = document.querySelector('.tab-pane.active') || document.body;
        parent.prepend(errorEl);
    }
    errorEl.innerHTML = `
        <i class="bi bi-exclamation-triangle-fill me-2"></i>${message}
        <button type="button" class="btn-close" onclick="hideError('${elementId}')"></button>
    `;
    errorEl.style.display = 'block';
}

function hideError(elementId) {
    const errorEl = document.getElementById(elementId);
    if (errorEl) errorEl.style.display = 'none';
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
