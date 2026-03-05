<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Annotate - Attribution Labelling</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://unpkg.com/wavesurfer.js@7"></script>
    <script src="https://unpkg.com/wavesurfer.js@7/dist/plugins/spectrogram.min.js"></script>
    <style>
        .img-fluid {
            max-height: 250px; /* Limit image height */
            width: 100%;
            object-fit: contain;
            background: #f8f9fa;
        }
        .detector-col {
            border-left: 1px solid #dee2e6;
        }
        .detector-col:first-child {
            border-left: none;
        }
        textarea {
            resize: vertical;
            min-height: 80px;
        }
        .metadata-badge {
            font-size: 0.85em;
            margin-right: 0.5em;
            margin-bottom: 0.5em;
        }
        #audioPlayer {
            width: 100%;
            margin-bottom: 1rem;
        }
        
        /* Ensure WaveSurfer canvas scales */
        #waveform > div, #waveform canvas {
            height: 100% !important;
            width: 100% !important;
        }
        
        /* Layout for Sticky Footer (at bottom of content, not viewport overlay) */
        html, body {
            height: 100%;
        }
        
        body {
            display: flex;
            flex-direction: column;
            padding-bottom: 0; 
        }
        
        .container-fluid {
            flex: 1 0 auto;
        }
        
        .nav-buttons {
            flex-shrink: 0;
            background: white;
            padding: 1rem;
            border-top: 1px solid #dee2e6;
            z-index: 1000;
        }
    </style>
</head>
<body class="bg-light">

<?php include 'nav.php'; ?>

<div class="container-fluid p-4">
    <!-- File Navigation Bar -->
    <div class="d-flex justify-content-between align-items-center mb-3 bg-white p-2 rounded shadow-sm border">
        
        <div class="d-flex align-items-center">
             <span id="fileCounter" class="me-3 fw-bold text-muted">Loading...</span>
             <select id="jumpToCategory" class="form-select form-select-sm d-inline-block w-auto" style="max-width: 300px;" onchange="jumpToCategory(this.value)">
                 <option value="">Jump to Category...</option>
            </select>
        </div>
        
    </div>

    <!-- Metadata Panel -->
    <div class="card mb-3">
        <div class="card-header bg-light">
            <strong>File ID: <span id="metaFileID"></span></strong>
        </div>
        <div class="card-body py-2">
            <div id="metadataContainer" class="d-flex flex-wrap">
                <!-- Badges populated by JS -->
            </div>
        </div>
    </div>

    <!-- Audio Player -->
    <div id="stickyAudioPlayer" class="bg-dark rounded p-3 mb-4 sticky-top" style="z-index: 1020;">
        <!-- <label class="text-white small mb-1">Audio Player</label> -->
        <div id="waveform" class="mb-2 bg-light rounded opacity-75"></div>
        <div id="spectrogram" class="rounded"></div>
        
        <!-- Time Bar Control -->
        <div class="position-relative mb-5">
             <datalist id="tickmarks"></datalist>
             
             <!-- Custom Visual Ticks -->
             <div id="visualTicks" class="position-absolute w-100 start-0 pointer-events-none" style="top: 0px; height: 30px; z-index: 1;"></div>
        </div>
        
        <div class="d-flex justify-content-between align-items-center position-relative">
            
            <div class="position-absolute start-0 ms-2 text-white font-monospace small">
                <!-- Time Display -->
                <span id="currentTimeLabel">0:00.00</span> / <span id="totalTimeLabel">0:00.00</span>
            </div>

            <div class="text-center w-100">
                <button class="btn btn-outline-light btn-sm me-2" onclick="skip(-1)" title="Back 1s">« 1s</button>
                <button class="btn btn-light btn-sm px-4" id="playPauseBtn" onclick="togglePlay()">
                    Play
                </button>
                <button class="btn btn-outline-light btn-sm ms-2" onclick="skip(1)" title="Forward 1s">1s »</button>
            </div>
            
        </div>
    </div>

    <!-- General Comment -->
    <div class="mb-3">
        <label for="generalComment" class="form-label fw-bold">General Comment / Observations</label>
        <textarea class="form-control" id="generalComment" placeholder="Any general observations about this sample..."></textarea>
    </div>

    <!-- Combined Plot -->
    <div class="card mb-4 shadow-sm">
        <div class="card-header fw-bold bg-white">
            Combined Attribution Analysis
        </div>
        <div class="card-body p-2">
             <div id="combinedPlot" style="height: 450px; width: 100%;"></div>
        </div>
    </div>

    <!-- Detectors Grid -->
    <div class="row mb-3">
        <!-- AASIST -->
        <div class="col-12 detector-row mb-4 bg-white p-3 rounded shadow-sm">
            <h5 class="border-bottom pb-2">AASIST</h5>
            <div class="row">
                <div class="col-12 bg-light rounded py-2 mb-3">
                     <div id="plotAASIST"></div>
                </div>
                <div class="col-12">
                    <div class="mb-2 fw-bold" title="Score Interpretation:&#013;Dataset Range: -18.33 to 43.81 (scale is nor linear!)&#013;&#013;> More Positive: Higher confidence in SPOOF (Fake)&#013;< More Negative: Higher confidence in BONAFIDE (Real)&#013;~ Between: Low confidence / Uncertain (the center is NOT 0)" style="cursor: help; text-decoration: underline dotted;">
                        Score: <span id="scoreAASIST">-</span>
                        <div id="infoAASIST" class="d-inline-block ms-2"></div>
                    </div>

                    <div class="mb-2">
                        <label class="form-label small fw-bold">Primary cue type (top segment) <span class="text-danger">*</span>:</label> 
                        <span id="regionStatusAASIST" class="badge bg-light text-dark border ms-2">No region</span>
                        <select class="form-select form-select-sm" id="cueAASIST">
                            <option value="">-- Select --</option>
                            <option value="glitch">Local glitch / artifact (clicks, buzz, robotic burst)</option>
                            <option value="phoneme">Phoneme content / articulation (vowels, consonant bursts)</option>
                            <option value="transition">Voiced–unvoiced transition (onset/offset, releases)</option>
                            <option value="silence">Silence / pause region (incl. inter-word gaps)</option>
                            <option value="breath">Breath / aspiration</option>
                            <option value="noise">Channel/codec/noise floor</option>
                            <option value="spectral">Spectral artifact (missing bands, cutoffs)</option>
                            <option value="unclear">Unclear / diffuse</option>
                        </select>
                    </div>

                    <div class="row mb-2">
                        <div class="col-8">
                             <label class="form-label small fw-bold mb-0">Locality (1=Spread, 5=Spiky):</label>
                             <div class="d-flex align-items-center">
                                 <span class="small me-2 text-muted">Spread</span>
                                 <input type="range" class="form-range" min="1" max="5" step="1" id="localityAASIST" oninput="document.getElementById('locValAASIST').textContent=this.value">
                                 <span class="small ms-2 text-muted">Spiky</span>
                             </div>
                             <div class="text-center small text-primary fw-bold" id="locValAASIST">-</div>
                        </div>
                        <div class="col-4 d-flex align-items-center">
                            <div class="form-check mt-3">
                                <input class="form-check-input" type="checkbox" id="audibleAASIST">
                                <label class="form-check-label small fw-bold" for="audibleAASIST">
                                    Is the main/primary cue audible?
                                </label>
                            </div>
                        </div>
                    </div>

                    <textarea class="form-control" id="commentAASIST" rows="3" placeholder="Comment on AASIST attribution..."></textarea>
                </div>
            </div>
        </div>

        <!-- CAMHFA -->
        <div class="col-12 detector-row mb-4 bg-white p-3 rounded shadow-sm">
            <h5 class="border-bottom pb-2">CAMHFA</h5>
            <div class="row">
                <div class="col-12 bg-light rounded py-2 mb-3">
                     <div id="plotCAMHFA"></div>
                </div>
                <div class="col-12">
                    <div class="mb-2 fw-bold" title="Score Interpretation:&#013;Dataset Range: -325.58 to 271.37 (scale is nor linear!)&#013;&#013;> More Positive: Higher confidence in SPOOF (Fake)&#013;< More Negative: Higher confidence in BONAFIDE (Real)&#013;~ Between: Low confidence / Uncertain (the center is NOT 0)" style="cursor: help; text-decoration: underline dotted;">
                        Score: <span id="scoreCAMHFA">-</span>
                        <div id="infoCAMHFA" class="d-inline-block ms-2"></div>
                    </div>

                    <div class="mb-2">
                        <label class="form-label small fw-bold">Primary cue type (top segment) <span class="text-danger">*</span>:</label> 
                        <span id="regionStatusCAMHFA" class="badge bg-light text-dark border ms-2">No region</span>
                        <select class="form-select form-select-sm" id="cueCAMHFA">
                            <option value="">-- Select --</option>
                            <option value="glitch">Local glitch / artifact (clicks, buzz, robotic burst)</option>
                            <option value="phoneme">Phoneme content / articulation (vowels, consonant bursts)</option>
                            <option value="transition">Voiced–unvoiced transition (onset/offset, releases)</option>
                            <option value="silence">Silence / pause region (incl. inter-word gaps)</option>
                            <option value="breath">Breath / aspiration</option>
                            <option value="noise">Channel/codec/noise floor</option>
                            <option value="spectral">Spectral artifact (missing bands, cutoffs)</option>
                            <option value="unclear">Unclear / diffuse</option>
                        </select>
                    </div>

                    <div class="row mb-2">
                        <div class="col-8">
                             <label class="form-label small fw-bold mb-0">Locality (1=Spread, 5=Spiky):</label>
                             <div class="d-flex align-items-center">
                                 <span class="small me-2 text-muted">Spread</span>
                                 <input type="range" class="form-range" min="1" max="5" step="1" id="localityCAMHFA" oninput="document.getElementById('locValCAMHFA').textContent=this.value">
                                 <span class="small ms-2 text-muted">Spiky</span>
                             </div>
                             <div class="text-center small text-primary fw-bold" id="locValCAMHFA">-</div>
                        </div>
                        <div class="col-4 d-flex align-items-center">
                            <div class="form-check mt-3">
                                <input class="form-check-input" type="checkbox" id="audibleCAMHFA">
                                <label class="form-check-label small fw-bold" for="audibleCAMHFA">
                                    Is the main/primary cue audible?
                                </label>
                            </div>
                        </div>
                    </div>

                    <textarea class="form-control" id="commentCAMHFA" rows="3" placeholder="Comment on CAMHFA attribution..."></textarea>
                </div>
            </div>
        </div>

        <!-- SLS -->
        <div class="col-12 detector-row mb-4 bg-white p-3 rounded shadow-sm">
            <h5 class="border-bottom pb-2">SLS</h5>
            <div class="row">
                <div class="col-12 bg-light rounded py-2 mb-3">
                     <div id="plotSLS"></div>
                </div>
                <div class="col-12">
                    <div class="mb-2 fw-bold" title="Score Interpretation:&#013;Dataset Range: -131.39 to 177.57 (scale is nor linear!)&#013;&#013;> More Positive: Higher confidence in SPOOF (Fake)&#013;< More Negative: Higher confidence in BONAFIDE (Real)&#013;~ Between: Low confidence / Uncertain (the center is NOT 0)" style="cursor: help; text-decoration: underline dotted;">
                        Score: <span id="scoreSLS">-</span>
                        <div id="infoSLS" class="d-inline-block ms-2"></div>
                    </div>

                    <div class="mb-2">
                        <label class="form-label small fw-bold">Primary cue type (top segment) <span class="text-danger">*</span>:</label> 
                        <span id="regionStatusSLS" class="badge bg-light text-dark border ms-2">No region</span>
                        <select class="form-select form-select-sm" id="cueSLS">
                            <option value="">-- Select --</option>
                            <option value="glitch">Local glitch / artifact (clicks, buzz, robotic burst)</option>
                            <option value="phoneme">Phoneme content / articulation (vowels, consonant bursts)</option>
                            <option value="transition">Voiced–unvoiced transition (onset/offset, releases)</option>
                            <option value="silence">Silence / pause region (incl. inter-word gaps)</option>
                            <option value="breath">Breath / aspiration</option>
                            <option value="noise">Channel/codec/noise floor</option>
                            <option value="spectral">Spectral artifact (missing bands, cutoffs)</option>
                            <option value="unclear">Unclear / diffuse</option>
                        </select>
                    </div>

                    <div class="row mb-2">
                        <div class="col-8">
                             <label class="form-label small fw-bold mb-0">Locality (1=Spread, 5=Spiky):</label>
                             <div class="d-flex align-items-center">
                                 <span class="small me-2 text-muted">Spread</span>
                                 <input type="range" class="form-range" min="1" max="5" step="1" id="localitySLS" oninput="document.getElementById('locValSLS').textContent=this.value">
                                 <span class="small ms-2 text-muted">Spiky</span>
                             </div>
                             <div class="text-center small text-primary fw-bold" id="locValSLS">-</div>
                        </div>
                        <div class="col-4 d-flex align-items-center">
                            <div class="form-check mt-3">
                                <input class="form-check-input" type="checkbox" id="audibleSLS">
                                <label class="form-check-label small fw-bold" for="audibleSLS">
                                    Is the main/primary cue audible?
                                </label>
                            </div>
                        </div>
                    </div>

                    <textarea class="form-control" id="commentSLS" rows="3" placeholder="Comment on SLS attribution..."></textarea>
                </div>
            </div>
        </div>
    </div>

    <!-- Similarity/Disparity -->
    <div class="mb-3">
        <label for="simComment" class="form-label fw-bold">Similarity / Disparity across detectors <span class="text-danger">*</span></label>
        <textarea class="form-control" id="simComment" placeholder="Compare the attributions. Do they focus on similar regions?"></textarea>
    </div>

</div>

<!-- Fixed Footer Navigation -->
<div class="nav-buttons d-flex justify-content-between align-items-center shadow-sm">
    <button class="btn btn-secondary px-4" id="prevBtn" onclick="prevFile()">Previous</button>
    <span id="navInfo" class="text-muted fw-bold"></span>
    <button class="btn btn-primary px-4" id="nextBtn" onclick="nextFile()">Next</button>
</div>

<script>
    const urlParams = new URLSearchParams(window.location.search);
    const username = urlParams.get('user');
    // Get recording_index from URL, default to 0. Parse as integer.
    const urlIndex = parseInt(urlParams.get('recording_index')) || 0;

    if (!username) {
        alert("No user selected!");
        window.location.href = 'index.php';
    }

    document.getElementById('usernameVal').textContent = username;

    let allFiles = []; // Data from CSV
    let userAnnotations = {}; // Data from JSON
    let currentSelections = {}; // { aasist: {start, end}, ... }
    let currentIndex = urlIndex;
    let wavesurfer;

    // Load data on startup
    async function init() {
        try {
            // Slider Interaction
            const slider = document.getElementById('audioSeek');
            if(slider) {
                slider.oninput = function() {
                    if(wavesurfer) {
                        const duration = wavesurfer.getDuration();
                        if(duration > 0) {
                            wavesurfer.seekTo(this.value / 1000);
                        }
                    }
                };
            }

            // Fetch list of files
            const filesRes = await fetch('api.php?action=get_files');
            allFiles = await filesRes.json();

            populateCategories();

            const userRes = await fetch(`api.php?action=get_user&username=${encodeURIComponent(username)}`);
            userAnnotations = await userRes.json();

            // Load first file
            loadView(currentIndex);

        } catch (e) {
            console.error("Initialization failed", e);
            alert("Failed to load data.");
        }
    }

    
    function renderIndividualPlot(containerId, data, modelName, color, initialSelection) {
        document.getElementById(containerId).innerHTML = ''; // Clear
        const fileSr = data.sample_rate || 16000;
        
        // Use normalized attributions if available
        const y_data = data.attributions || data.attributions_raw; 
        
        if (!y_data) {
             document.getElementById(containerId).innerHTML = '<div class="alert alert-warning">No attribution data</div>';
             return;
        }

        const traces = [];

        // X Axis
        const x_attr = new Float32Array(y_data.length);
        const attr_duration = y_data.length * 320 / fileSr;
        if (y_data.length > 1) {
             const step = attr_duration / (y_data.length-1); 
             for(let i=0; i<y_data.length; i++) x_attr[i] = i * step;
        } else {
             x_attr[0] = 0;
        }

        // 1. Waveform (Background)
        if(data.waveform) {
             const count = data.waveform.length;
             const x_wave = new Float32Array(count);
             for(let i=0; i<count; i++) x_wave[i] = i / fileSr;
             
             traces.push({
                x: x_wave,
                y: data.waveform,
                type: 'scatter', 
                mode: 'lines',
                name: 'Waveform',
                line: {color: 'rgba(0, 0, 0, 0.20)', width: 1},
                hoverinfo: 'none'
            });
        }
        
        // 2. Attribution
        // Split into positive (Spoof) and negative (Bonafide)
        const y_spoof = y_data.map(v => Math.max(v, 0));
        const y_bonafide = y_data.map(v => Math.min(v, 0));

        traces.push({
            x: x_attr,
            y: y_spoof,
            type: 'scatter',
            mode: 'lines',
            name: 'Spoof Evidence (>0)',
            line: { color: 'rgba(220, 53, 69, 1.0)', width: 1.5 }, 
            // fill: 'tozeroy',
            opacity: 0.9
        });
        
        traces.push({
            x: x_attr,
            y: y_bonafide,
            type: 'scatter',
            mode: 'lines',
            name: 'Bonafide Evidence (<0)',
            line: { color: 'rgba(0.1725, 0.6274, 0.1725, 1.0)', width: 1.5 },
            opacity: 0.9
        });

        // 3. Median Trend (normalized if possible)
        const y_trend = data.median_trend || data.median_trend_raw;
        if (y_trend) {
            const x_trend = (y_trend.length === y_data.length) ? x_attr : new Float32Array(y_trend.length);
            if (x_trend !== x_attr) {
                 for(let i=0; i<y_trend.length; i++) x_trend[i] = i * 320 / fileSr;
            }

            traces.push({
                x: x_trend,
                y: y_trend,
                type: 'scatter',
                mode: 'lines',
                name: 'Trend',
                line: { color: 'rgb(98, 0, 131)', width: 2, dash: 'solid' }, // Made trend solid and bolder for visibility
            });
        }

        const layout = {
            margin: {t: 30, b: 30, l: 40, r: 10},
            showlegend: true,
            legend: {orientation: "h", yanchor: "top", y: 1.1, xanchor: "left", x: 0},
            xaxis: {title: 'Time (s)'},
            yaxis: {title: 'Attribution (Normalized)', fixedrange: true}, // Fixed Y-axis
            autosize: true,
            dragmode: 'select', // Default to select
            selectdirection: 'h', // Restrict to horizontal (time) axis
            height: 300,
            title: { text: modelName.toUpperCase(), font: {size: 14} },
            shapes: []
        };

        // Add selection shape if exists
        if (initialSelection && initialSelection.start !== undefined && initialSelection.end !== undefined) {
            layout.shapes.push({
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: initialSelection.start,
                x1: initialSelection.end,
                y0: 0,
                y1: 1,
                fillcolor: 'rgba(255, 255, 0, 0.2)',
                line: { width: 0 }
            });
        }
        
        // Configure modebar with Zoom, Pan, Select
        const config = {
            responsive: true, 
            displayModeBar: true,
            modeBarButtons: [[ 'zoom2d', 'pan2d', 'select2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d' ]]
        };

        Plotly.newPlot(containerId, traces, layout, config).then(gd => {
            // Handle Selection
            gd.on('plotly_selected', function(eventData) {
                let sel = null;
                if (eventData && eventData.range) {
                    const range = eventData.range.x; // [min, max]
                    if(range && range.length === 2) {
                        sel = { start: range[0], end: range[1] };
                    }
                } else if(eventData && eventData.points && eventData.points.length > 0) {
                     const xs = eventData.points.map(p => p.x);
                     sel = { start: Math.min(...xs), end: Math.max(...xs) };
                }

                // Only update if we have a valid selection (prevent accidental unselect)
                if (sel) {
                    currentSelections[modelName.toLowerCase()] = sel;
                    updateSelectionUI(modelName, sel);

                    // Update Visual Shape immediately to "lock" it and remove the native selection box
                    const newShapes = [];
                    newShapes.push({
                        type: 'rect',
                        xref: 'x',
                        yref: 'paper',
                        x0: sel.start,
                        x1: sel.end,
                        y0: 0,
                        y1: 1,
                        fillcolor: 'rgba(255, 255, 0, 0.2)',
                        line: { width: 0 }
                    });
                    
                    // Use relayout to update shapes. This often clears the active selection box interaction.
                    Plotly.relayout(containerId, { shapes: newShapes });
                }
            });
            
            // Handle Double Click to reset (Disabled to prevent unselection)
            /*
            gd.on('plotly_doubleclick', function() {
                 currentSelections[modelName.toLowerCase()] = null;
                 updateSelectionUI(modelName, null);
                 Plotly.relayout(containerId, { shapes: [] });
            });
            */
        });

        // Initialize UI with current selection
        updateSelectionUI(modelName, initialSelection);
    }

    function updateSelectionUI(modelName, selection) {
        const uiId = `selInfo${modelName.toUpperCase()}`;
        const badgeId = `regionStatus${modelName.toUpperCase()}`; // Defines the badge ID
        
        // 1. Update text below plot
        let el = document.getElementById(uiId);
        if (!el) {
             const container = document.getElementById(`plot${modelName.toUpperCase()}`).parentNode;
             el = document.createElement('div');
             el.id = uiId;
             el.className = 'small text-muted mt-1 text-center';
             container.appendChild(el);
        }

        // 2. Update badge next to select input
        const badge = document.getElementById(badgeId);

        if (selection && selection.start !== undefined) {
            el.innerHTML = `<strong>Selected CUE Region:</strong> ${selection.start.toFixed(3)}s — ${selection.end.toFixed(3)}s`;
            el.style.color = 'black';
            
            if (badge) {
                badge.className = 'badge bg-success border ms-2';
                badge.textContent = 'Region Selected';
            }
        } else {
            el.innerHTML = `<em>No region selected. Use "Box Select" tool to mark the primary cue.</em>`;
            el.style.color = '#6c757d'; // secondary
            
            if (badge) {
                badge.className = 'badge bg-danger ms-2 fs-6';
                badge.textContent = 'No region';
            }
        }
    }

    
    async function renderCombinedPlot(fileId) {
        const containerId = 'combinedPlot';
        document.getElementById(containerId).innerHTML = 'Loading combined plot...';
        
        const models = ['aasist', 'camhfa', 'sls'];
        const colors = {
            'aasist': 'rgba(220, 53, 69, 0.8)',   // Danger Red
            'camhfa': 'rgba(25, 135, 84, 0.8)',   // Success Green
            'sls': 'rgba(13, 110, 253, 0.8)'      // Primary Blue
        };
        
        try {
            // Fetch all concurrently
            const results = await Promise.allSettled(
                models.map(m => fetch(`outputs/IG/${fileId}_${m}_diff_baseline.json`).then(r => {
                    if(!r.ok) throw new Error(r.statusText);
                    return r.json();
                }))
            );

            const traces = [];
            let waveform = null;
            let sr = 16000;
            
            // Process results
            results.forEach((res, idx) => {
                const modelName = models[idx];
                if (res.status === 'fulfilled') {
                    const data = res.value;
                    const fileSr = data.sample_rate || 16000;

                    if (!waveform && data.waveform) {
                        waveform = data.waveform;
                        sr = fileSr;
                    }
                    
                    // Add IG trace
                    // Check if 'attributions_raw' exists, otherwise use 'attributions'
                    const y_data = data.attributions_raw || data.attributions;

                    // Generate X axis for attributions (frame hop = 320)
                    // Match Python's np.linspace(0, len(attr) * 320 / sr, len(attr)) logic
                    const x_attr = new Float32Array(y_data.length);
                    const attr_duration = y_data.length * 320 / fileSr;
                    if (y_data.length > 1) {
                         const step = attr_duration / (y_data.length-1); // linspace behavior
                         for(let i=0; i<y_data.length; i++) x_attr[i] = i * step;
                    } else {
                         x_attr[0] = 0;
                    }
                    
                    traces.push({
                        x: x_attr,
                        y: y_data,
                        type: 'scatter',
                        mode: 'lines',
                        name: modelName.toUpperCase(),
                        line: { color: colors[modelName], width: 1.5 },
                        opacity: 0.9
                    });

                    // Add Median Trend (optional, hidden by default)
                    const y_trend = data.median_trend_raw || data.median_trend;
                    if (y_trend) {
                        const x_trend = (y_trend.length === y_data.length) ? x_attr : new Float32Array(y_trend.length);
                        if (x_trend !== x_attr) {
                             for(let i=0; i<y_trend.length; i++) x_trend[i] = i * 320 / fileSr;
                        }

                        traces.push({
                            x: x_trend,
                            y: y_trend,
                            type: 'scatter',
                            mode: 'lines',
                            name: `${modelName.toUpperCase()} Trend`,
                            line: { color: colors[modelName], width: 2.5, dash: 'dot' },
                            visible: 'legendonly'
                        });
                    }

                    // Render Individual Plot
                    const modelKey = modelName.toLowerCase();
                    renderIndividualPlot(`plot${modelName.toUpperCase()}`, data, modelName, colors[modelName], currentSelections[modelKey]);
                } else {
                    document.getElementById(`plot${modelName.toUpperCase()}`).innerHTML = '<div class="alert alert-warning">No data</div>';
                }
            });

            if (waveform) {
                // Generate X axis
                const count = waveform.length;
                const x_wave = new Float32Array(count);
                for(let i=0; i<count-320; i++) x_wave[i] = i / sr;

                // Add Waveform trace (background)
                traces.unshift({
                    x: x_wave,
                    y: waveform,
                    type: 'scatter', 
                    mode: 'lines',
                    name: 'Waveform',
                    line: {color: 'rgba(0,0,0,0.15)', width: 1},
                    hoverinfo: 'none'
                });
                
                const layout = {
                    title: { text: `Combined IG Attribution - ${fileId}`, font: {size: 14} },
                    margin: {t: 40, b: 40, l: 50, r: 20},
                    showlegend: true,
                    // Move legend to top-left to avoid conflict with modebar (top-right)
                    legend: {orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0},
                    xaxis: {title: 'Time (s)'},
                    yaxis: {title: 'Raw Attribution'},
                    autosize: true,
                    // Enable scroll/zoom better
                    dragmode: 'zoom'
                };
                
                const config = {
                    responsive: true, 
                    displayModeBar: true,
                    modeBarButtonsToRemove: ['lasso2d', 'select2d']
                };

                // Clear "Loading..." message
                document.getElementById(containerId).innerHTML = '';
                Plotly.newPlot(containerId, traces, layout, config);
            } else {
                 document.getElementById(containerId).innerHTML = '<div class="alert alert-warning">No attribution data found for any model.</div>';
            }

        } catch (e) {
            console.error("Combined plot error", e);
            document.getElementById(containerId).innerHTML = '<div class="text-danger">Failed to load combined plot.</div>';
        }
    }

    function initializeWaveSurfer(url) {
        if (wavesurfer) {
            wavesurfer.destroy();
        }

        wavesurfer = WaveSurfer.create({
            container: '#waveform',
            url: url,
            waveColor: 'rgb(50, 50, 50)',
            progressColor: 'rgb(100, 0, 150)',
            height: 60,
            barWidth: 2,
            cursorWidth: 1,
            plugins: [
                WaveSurfer.Spectrogram.create({
                    container: '#spectrogram',
                    labels: true,
                    height: 120,
                    labelsColor: 'white'
                })
            ]
        });

        // Event listeners for UI updates
        wavesurfer.on('ready', () => {
            updateTimeDisplay();
            generateTicks();
        });
        wavesurfer.on('timeupdate', updateTimeDisplay);
        wavesurfer.on('audioprocess', updateTimeDisplay); // For compatibility
        wavesurfer.on('interaction', updateTimeDisplay);  // Updates when user clicks/drags
        wavesurfer.on('finish', () => {
             const btn = document.getElementById('playPauseBtn');
             if(btn) btn.textContent = 'Play';
        });
        wavesurfer.on('play', () => {
             const btn = document.getElementById('playPauseBtn');
             if(btn) btn.textContent = 'Pause';
        });
        wavesurfer.on('pause', () => {
             const btn = document.getElementById('playPauseBtn');
             if(btn) btn.textContent = 'Play';
        });
    }

    async function loadView(index) {
        if (allFiles.length === 0) return;
        
        // Clamp index
        if (index < 0) index = 0;
        if (index >= allFiles.length) index = allFiles.length - 1;
        currentIndex = index;

        // Update URL
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('recording_index', currentIndex);
        window.history.pushState({}, '', newUrl);

        const fileData = allFiles[index];
        const fileId = fileData['FileID'];
        
        // --- 1. Top Metadata ---
        document.getElementById('fileCounter').textContent = `Record ${index + 1} of ${allFiles.length}`;
        document.getElementById('navInfo').textContent = `${fileId} (${index + 1}/${allFiles.length})`;
        
        // --- PRE-LOAD ANNOTATIONS (Moved up to be available for plots) ---
        const ann = userAnnotations[fileId] || {};
        currentSelections = {
            aasist: ann.aasist_range || null,
            camhfa: ann.camhfa_range || null,
            sls: ann.sls_range || null
        };

        document.getElementById('metaFileID').textContent = fileId;

        const metaContainer = document.getElementById('metadataContainer');
        metaContainer.innerHTML = '';
        
        // Helper to create badges
        const createBadge = (key, value) => {
            const badge = document.createElement('span');
            badge.className = 'badge bg-secondary metadata-badge me-2';
            badge.innerHTML = `<span class="fw-light">${key}:</span> ${value}`;
            return badge;
        };

        // Row 1: Label, Attack, Type, Algorithm, Codec
        const row1 = document.createElement('div');
        row1.className = 'mb-2 w-100';
        
        if (fileData['Label']) row1.appendChild(createBadge('Label', fileData['Label']));
        
        if (fileData['Attack']) row1.appendChild(createBadge('Attack', fileData['Attack']));
        if (fileData['Attack Type']) row1.appendChild(createBadge('Type', fileData['Attack Type']));
        if (fileData['Algorithm']) row1.appendChild(createBadge('Algorithm', fileData['Algorithm']));

        if (fileData['Codec']) row1.appendChild(createBadge('Codec', fileData['Codec']));
        metaContainer.appendChild(row1);

        // Row 2: Speaker, Gender
        const row2 = document.createElement('div');
        row2.className = 'mb-2 w-100';
        if (fileData['Speaker']) row2.appendChild(createBadge('Speaker', fileData['Speaker']));
        if (fileData['Gender']) row2.appendChild(createBadge('Gender', fileData['Gender']));
        metaContainer.appendChild(row2);

        // Row 3: Selection Reason
        const row3 = document.createElement('div');
        row3.className = 'mb-2 w-100'; 
        if (fileData['SelectionReason']) {
            const badge = createBadge('Selection Reason', fileData['SelectionReason']);
            badge.style.whiteSpace = 'normal';
            badge.style.textAlign = 'left';
            row3.appendChild(badge);
        }
        metaContainer.appendChild(row3);

        // --- 2. Media ---
        // Load audio
        initializeWaveSurfer(`recordings_48k/${fileId}.flac`);

        // Render Interactive Plots
        await renderCombinedPlot(fileId);
        
        // Individual plots (kept for reference, or could be removed if redundant)
        // document.getElementById('imgAASIST').src = `outputs/IG/${fileId}_aasist_diff_baseline.png`;
        // document.getElementById('imgCAMHFA').src = `outputs/IG/${fileId}_camhfa_diff_baseline.png`;
        // document.getElementById('imgSLS').src = `outputs/IG/${fileId}_sls_diff_baseline.png`;

        // --- 3. Scores ---
        document.getElementById('scoreAASIST').textContent = parseFloat(fileData['AASIST_Score']).toFixed(4);
        document.getElementById('scoreCAMHFA').textContent = parseFloat(fileData['CAMHFA_Score']).toFixed(4);
        document.getElementById('scoreSLS').textContent = parseFloat(fileData['SLS_Score']).toFixed(4);

        // --- Render Info Badges ---
        const reason = fileData['SelectionReason'] || '';
        
        const renderInfo = (elementId) => {
            const el = document.getElementById(elementId);
            if (!reason) { el.innerHTML = ''; return; }
            
            // Logic to color code
            let badgeClass = 'bg-secondary';
            if (reason.includes('Correct') || reason.includes('CR')) {
                badgeClass = 'bg-success';
            } else if (reason.includes('Wrong') || reason.includes('CW') || reason.includes('Worst')) {
                badgeClass = 'bg-danger';
            } else if (reason.includes('Mid')) {
                badgeClass = 'bg-warning text-dark';
            }
            
            el.innerHTML = `<span class="badge ${badgeClass}">${reason}</span>`;
        };
        
        renderInfo('infoAASIST');
        renderInfo('infoCAMHFA');
        renderInfo('infoSLS');


        // --- 4. Load Existing Annotations ---
        /* const ann ... moved up */
        document.getElementById('generalComment').value = ann.general || '';
        
        /* Selections restored above */
        
        // AASIST
        document.getElementById('commentAASIST').value = ann.aasist || '';
        document.getElementById('cueAASIST').value = ann.aasist_cue || '';
        document.getElementById('localityAASIST').value = ann.aasist_loc || 3;
        document.getElementById('locValAASIST').textContent = ann.aasist_loc || '-';
        document.getElementById('audibleAASIST').checked = !!ann.aasist_audible;

        // CAMHFA
        document.getElementById('commentCAMHFA').value = ann.camhfa || '';
        document.getElementById('cueCAMHFA').value = ann.camhfa_cue || '';
        document.getElementById('localityCAMHFA').value = ann.camhfa_loc || 3;
        document.getElementById('locValCAMHFA').textContent = ann.camhfa_loc || '-';
        document.getElementById('audibleCAMHFA').checked = !!ann.camhfa_audible;

        // SLS
        document.getElementById('commentSLS').value = ann.sls || '';
        document.getElementById('cueSLS').value = ann.sls_cue || '';
        document.getElementById('localitySLS').value = ann.sls_loc || 3;
        document.getElementById('locValSLS').textContent = ann.sls_loc || '-';
        document.getElementById('audibleSLS').checked = !!ann.sls_audible;
        
        document.getElementById('simComment').value = ann.similarity || '';

        // --- 5. Button States ---
        document.getElementById('prevBtn').disabled = (index === 0);
        document.getElementById('nextBtn').disabled = (index === allFiles.length - 1);
    }

    async function saveCurrent() {
        if (allFiles.length === 0) return;
        
        const fileId = allFiles[currentIndex]['FileID'];

        // Collect data
        const data = {
            general: document.getElementById('generalComment').value,
            
            aasist: document.getElementById('commentAASIST').value,
            aasist_cue: document.getElementById('cueAASIST').value,
            aasist_loc: document.getElementById('localityAASIST').value,
            aasist_audible: document.getElementById('audibleAASIST').checked,
            
            camhfa: document.getElementById('commentCAMHFA').value,
            camhfa_cue: document.getElementById('cueCAMHFA').value,
            camhfa_loc: document.getElementById('localityCAMHFA').value,
            camhfa_audible: document.getElementById('audibleCAMHFA').checked,

            sls: document.getElementById('commentSLS').value,
            sls_cue: document.getElementById('cueSLS').value,
            sls_loc: document.getElementById('localitySLS').value,
            sls_audible: document.getElementById('audibleSLS').checked,

            // Saved ranges
            aasist_range: currentSelections['aasist'] || null,
            camhfa_range: currentSelections['camhfa'] || null,
            sls_range: currentSelections['sls'] || null,

            similarity: document.getElementById('simComment').value
        };

        // Update local state
        userAnnotations[fileId] = data;

        // Send to server
        // Note: For a "real-time" feel we send the WHOLE state or just PATCH. 
        // Our PHP 'save_user' accepts the full object structure for the user file.
        // It might be heavy if the file gets huge, but for 100 records it is fine.
        
        try {
            await fetch('api.php?action=save_user', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: username,
                    data: userAnnotations
                })
            });
        } catch (e) {
            console.error("Save failed", e);
            alert("Warning: Failed to save changes!");
        }
    }

    async function nextFile() {
        if (!validateForm()) return;
        if(document.activeElement) document.activeElement.blur();
        await saveCurrent();
        await loadView(currentIndex + 1);
    }

    function validateForm() {
        const models = ['aasist', 'camhfa', 'sls'];
        let errors = [];

        models.forEach(m => {
            const M = m.toUpperCase();
            // Check Selection
            if (!currentSelections[m]) {
                errors.push(`${M}: Region selection is required.`);
            }
            // Check Cue
            const cue = document.getElementById(`cue${M}`).value;
            if (!cue) {
                 errors.push(`${M}: Primary cue type is required.`);
            }
        });

        // Check Similarity
        const sim = document.getElementById('simComment').value.trim();
        if (!sim) {
            errors.push("Similarity / Disparity comment is required.");
        }

        if (errors.length > 0) {
            alert("Please complete the required fields:\n\n- " + errors.join("\n- "));
            return false;
        }
        return true;
    }

    async function prevFile() {
        if(document.activeElement) document.activeElement.blur();
        await saveCurrent();
        await loadView(currentIndex - 1);
    }

    function populateCategories() {
        const categories = {};
        allFiles.forEach((file, idx) => {
            // Trim whitespace to avoid duplicates
            const reason = (file['SelectionReason'] || '').trim();
            if (reason && !categories.hasOwnProperty(reason)) {
                categories[reason] = idx;
            }
        });

        const select = document.getElementById('jumpToCategory');
        // Keep default option
        select.innerHTML = '<option value="">Jump to Category...</option>';
        
        // Sort categories alphabetically
        Object.keys(categories).sort().forEach(cat => {
            const opt = document.createElement('option');
            opt.value = categories[cat]; // Store index as value
            // Truncate if too long
            opt.textContent = cat.length > 60 ? cat.substring(0, 60) + '...' : cat;
            select.appendChild(opt);
        });
    }

    async function jumpToCategory(indexStr) {
        if (indexStr === "") return;
        const index = parseInt(indexStr);
        if (!isNaN(index)) {
             await saveCurrent();
             await loadView(index);
             // Reset dropdown
             document.getElementById('jumpToCategory').value = "";
             if(document.activeElement) document.activeElement.blur();
        }
    }

    function togglePlay() {
        if(wavesurfer) {
            wavesurfer.playPause();
        }
    }
    
    function skip(seconds) {
        if(wavesurfer) {
            wavesurfer.skip(seconds);
        }
    }

    function formatTime(seconds) {
        if(!seconds) return "0:00.00";
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 100);
        return `${minutes}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
    }

    function generateTicks() {
        if (!wavesurfer) return;
        const duration = wavesurfer.getDuration();
        const datalist = document.getElementById('tickmarks');
        const visualTicks = document.getElementById('visualTicks');
        
        if (duration <= 0) return;

        if (datalist) datalist.innerHTML = '';
        if (visualTicks) visualTicks.innerHTML = '';
        
        // Add ticks every 0.1 seconds
        const step = 0.1; 
        for(let t = 0; t <= duration + 0.05; t += step) {
            const pct = (t / duration) * 100;
            const val = (t / duration) * 1000;
            
            // 1. Native Datalist Point (for snapping)
            if (datalist) {
                const option = document.createElement('option');
                option.value = val;
                datalist.appendChild(option);
            }

            // 2. Visual Tick Mark
            if (visualTicks) {
                // Determine tick type
                const isFullSecond = (Math.abs(t - Math.round(t)) < 0.01);
                const isHalfSecond = (Math.abs((t % 1) - 0.5) < 0.01);
                
                // Tick line
                const tick = document.createElement('div');
                tick.className = 'position-absolute bg-white'; 
                
                tick.style.left = `${pct}%`;
                tick.style.width = '1px';
                tick.style.top = '0';
                tick.style.opacity = '0.8';

                if (isFullSecond) {
                    tick.style.height = '12px';
                    tick.style.width = '2px'; // Thicker for seconds
                } else if (isHalfSecond) {
                    tick.style.height = '8px';
                } else {
                    tick.style.height = '4px';
                    tick.style.opacity = '0.5';
                }

                visualTicks.appendChild(tick);
                
                // Label for full seconds and half seconds
                if (isFullSecond || isHalfSecond) {
                    const label = document.createElement('div');
                    label.className = 'position-absolute text-white small font-monospace';
                    label.style.left = `${pct}%`;
                    label.style.top = '14px';
                    label.style.transform = 'translateX(-50%)';
                    
                    if (isFullSecond) {
                        label.style.fontSize = '0.7em';
                        label.textContent = Math.round(t); 
                    } else {
                         // Half second
                        label.style.fontSize = '0.6em';
                        label.style.opacity = '0.7';
                        label.textContent = t.toFixed(1);
                    }
                    
                    visualTicks.appendChild(label);
                }
            }
        }
    }

    function updateTimeDisplay() {
        if (!wavesurfer) return;
        const curr = wavesurfer.getCurrentTime();
        const total = wavesurfer.getDuration();
        
        const currentLabel = document.getElementById('currentTimeLabel');
        const totalLabel = document.getElementById('totalTimeLabel');
        
        if(currentLabel) currentLabel.textContent = formatTime(curr);
        if(totalLabel) totalLabel.textContent = formatTime(total);
        
        // Update slider
        const slider = document.getElementById('audioSeek');
        if(slider && total > 0) {
           // Only update if not currently being dragged? 
           // HTML5 ranges don't have a simple 'isDragging' state without listeners, 
           // but updating .value usually works fine even during drag on modern browsers, 
           // or we can tolerate the jitter.
           // To be safe, we check if active element is the slider.
           if(document.activeElement !== slider) {
               slider.value = (curr / total) * 1000;
           }
        }
    }

    // Spacebar to play/pause
    document.addEventListener('keydown', function(event) {
        if (event.code === 'Space') {
            const active = document.activeElement;
            const tag = active.tagName.toLowerCase();
            const isSlider = (active.id === 'audioSeek');
            const player = document.getElementById('stickyAudioPlayer');
            const isMediaButton = (tag === 'button' && player && player.contains(active));

            // If we are in a text input or textarea, do nothing (let user type)
            // Unless it is the slider (which is technically an input)
            if (!isSlider && (tag === 'textarea' || tag === 'input')) {
                return; 
            }
            
            // If we are on a button that is NOT a media control (e.g. Next/Prev), do nothing (let user click)
            if (tag === 'button' && !isMediaButton) {
                return;
            }

            // Otherwise (Body, Slider, Media Buttons):
            // Prevent default action (Scroll or Button Click) and Toggle Play directly
            event.preventDefault();
            togglePlay();
        }
    });

    // Start
    init();

</script>

</body>
</html>
