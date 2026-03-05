<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Annotation Methodology</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .step-number {
            background-color: #0d6efd;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 10px;
        }
        .step-item {
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body class="bg-light">

<?php include 'nav.php'; ?>

<div class="container mt-4">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card shadow-sm mb-5">
                <div class="card-header bg-primary text-white">
                    <h3 class="mb-0">Annotation Methodology</h3>
                </div>
                <div class="card-body p-4">
                    <div class="alert alert-info mb-4" role="alert">
                        <h4 class="alert-heading">Welcome & Thank You!</h4>
                        <p>Thank you for taking the time to conduct this research. Your contribution is vital for interpreting deepfake detection models.</p>
                        <hr>
                        <p class="mb-2"><strong>Before you begin, please ensure the following:</strong></p>
                        <ul class="mb-0">
                            <li>Audio Equipment: Use <strong>quality headphones</strong> (ideally overhead/over-ear). Laptop speakers or cheap earbuds may miss critical details.</li>
                            <li>Environment: Find a <strong>silent room</strong> free from distractions and background noise.</li>
                            <li>Time: Ensure you have <strong>enough time</strong>. You can pause the annotations and come back to it later.</li>
                            <li>Browser: Use a modern web browser for the best experience with interactive plots and audio.</li>
                        </ul>
                    </div>

                    <p class="lead mb-4">Please follow this step-by-step process for each recording to ensure consistent and high-quality annotations.</p>

                    <div class="step-item d-flex">
                        <div class="step-number">1</div>
                        <div>
                            <h5>Initial Assessment</h5>
                            <p>Listen to the recording. If there is anything interesting straight away, write it in the first <strong>"General Comment / Observations"</strong> text field. Some guiding questions you can try to answer:</p>
                            <ul>
                                <li>Is there noticeable background noise or interference?</li>
                                <li>Is the speech understandable and clear?</li>
                                <li>Do you hear any unexpected sounds or anomalies?</li>
                                <li>Are there any sudden changes or interruptions?</li>
                                <li>Are there any other obvious issues with the recording quality?</li>
                            </ul>
                        </div>
                    </div>

                    <hr class="my-4">
                    <h5 class="text-muted mb-4">For Each Detector (AASIST, CAMHFA, SLS):</h5>

                    <div class="step-item d-flex">
                        <div class="step-number">2</div>
                        <div>
                            <h5>Localize Evidence</h5>
                            <p>Identify the top evidence segment (the highest peak or most prominent region in the IG curve) for the specific detector. Use your mouse on the plot to select this segment.</p>
                        </div>
                    </div>

                    <div class="step-item d-flex">
                        <div class="step-number">3</div>
                        <div>
                            <h5>Primary Cue Type</h5>
                            <p>Choose the <strong>Primary cue type</strong> for that top segment from the dropdown menu (e.g., Glitch, Phoneme, Silence, etc.).</p>
                        </div>
                    </div>

                    <div class="step-item d-flex">
                        <div class="step-number">4</div>
                        <div>
                            <h5>Rate Locality</h5>
                            <p>Rate the <strong>Locality/Spread</strong> of the IG attributions using the slider (1–5):</p>
                            <ul>
                                <li><strong>1 (Spread):</strong> Attribution is diffuse and spread across time.</li>
                                <li><strong>5 (Spiky):</strong> Attribution is highly localized to a specific moment.</li>
                            </ul>
                        </div>
                    </div>

                    <div class="step-item d-flex">
                        <div class="step-number">5</div>
                        <div>
                            <h5>Audibility</h5>
                            <p>Check the <strong>"Is the main/primary cue audible?"</strong> box if you can hear the specific artifact or cue identified by the IG curve.</p>
                        </div>
                    </div>

                    <div class="step-item d-flex">
                        <div class="step-number">6</div>
                        <div>
                            <h5>Short Comment</h5>
                            <p>Add a short comment describing what the IG curve actually localized (e.g., “buzz at onset,” “hard cut,” “excess silence,” “vowel transition”).</p>
                        </div>
                    </div>

                    <hr class="my-4">

                    <div class="step-item d-flex">
                        <div class="step-number">7</div>
                        <div>
                            <h5>Synthesis</h5>
                            <p>Finally, fill the last <strong>"Similarity / Disparity"</strong> text field. Summarize whether the different detectors focus on similar regions or if they are looking at completely different cues.</p>
                        </div>
                    </div>

                    <div class="mt-5 text-center">
                        <a href="index.php" class="btn btn-primary btn-lg">Back to Home/Start Labelling</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
