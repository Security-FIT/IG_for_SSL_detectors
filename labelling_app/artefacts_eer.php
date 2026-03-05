<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Artefact EER Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .table-responsive {
            max-height: 800px;
            overflow-y: auto;
        }
        thead th {
            position: sticky;
            top: 0;
            background-color: white;
            z-index: 1;
        }
    </style>
</head>
<body class="bg-light">

<?php include 'nav.php'; ?>

<div class="container mt-4">
    <h1 class="mb-4">Artefact EER Analysis</h1>
    
    <div class="row">
        <div class="col-md-5">
            <div class="card shadow-sm mb-4">
                <div class="card-header bg-primary text-white">
                    <h5 class="mb-0">Global EER (Bonafide vs Spoof) of the selected recordings</h5>
                </div>
                <div class="card-body p-0">
                    <div class="table-responsive">
                        <table class="table table-striped table-hover mb-0">
                            <thead class="table-light">
                                <tr>
                                    <th>Metric</th>
                                    <th>EER</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php
                                $global_file = 'outputs/artefact_eer_global.csv';
                                if (file_exists($global_file)) {
                                    $handle = fopen($global_file, "r");
                                    $header = fgetcsv($handle); // Skip header
                                    while (($data = fgetcsv($handle)) !== FALSE) {
                                        $metric = htmlspecialchars($data[0]);
                                        $eer = min(floatval($data[1]), 1-floatval($data[1])); // Ensure EER <= 0.5
                                        $color = $eer < 0.4 ? 'text-success fw-bold' : '';
                                        echo "<tr>
                                            <td>{$metric}</td>
                                            <td class='{$color}'>" . number_format($eer, 4) . "</td>
                                        </tr>";
                                    }
                                    fclose($handle);
                                } else {
                                    echo "<tr><td colspan='2'>Global CSV not found</td></tr>";
                                }
                                ?>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        
        <div class="col-md-7">
            <div class="card shadow-sm mb-4">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0">Pairwise EER (Selection Reasons)</h5>
                </div>
                <div class="card-body">
                    <p class="text-muted small">Showing pairs with EER < 0.30</p>
                    <div class="table-responsive">
                        <table class="table table-sm table-striped table-hover">
                            <thead class="table-light">
                                <tr>
                                    <th>Group 1</th>
                                    <th>Group 2</th>
                                    <th>Metric</th>
                                    <th>EER</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php
                                $pairwise_file = 'outputs/artefact_eer_pairwise.csv';
                                if (file_exists($pairwise_file)) {
                                    $rows = [];
                                    $handle = fopen($pairwise_file, "r");
                                    $header = fgetcsv($handle);
                                    while (($data = fgetcsv($handle)) !== FALSE) {
                                        $rows[] = [
                                            'g1' => $data[0],
                                            'g2' => $data[1],
                                            'metric' => $data[2],
                                            'eer' => floatval($data[3])
                                        ];
                                    }
                                    fclose($handle);

                                    // Sort by EER ascending
                                    usort($rows, function($a, $b) {
                                        return $a['eer'] <=> $b['eer'];
                                    });

                                    // Display top results (e.g., EER < 0.3)
                                    foreach ($rows as $row) {
                                        if ($row['eer'] > 0.3) continue; // Filter
                                        
                                        // Exclude Worst-Scored Spoof as requested
                                        if ($row['g1'] === 'Worst-Scored Spoof' || $row['g2'] === 'Worst-Scored Spoof') continue;

                                        $eer_val = number_format($row['eer'], 4);
                                        $highlight = $row['eer'] < 0.1 ? 'table-warning' : '';
                                        
                                        echo "<tr class='{$highlight}'>
                                            <td><small>{$row['g1']}</small></td>
                                            <td><small>{$row['g2']}</small></td>
                                            <td>{$row['metric']}</td>
                                            <td class='fw-bold'>{$eer_val}</td>
                                        </tr>";
                                    }
                                } else {
                                    echo "<tr><td colspan='4'>Pairwise CSV not found</td></tr>";
                                }
                                ?>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
