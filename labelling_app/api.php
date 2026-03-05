<?php
header('Content-Type: application/json');

$action = $_GET['action'] ?? '';
$base_csv = 'final_selection_132_details.csv';
$users_dir = 'users/';

if ($action === 'get_files') {
    // Load attacks metadata
    $attacks = [];
    if (file_exists('attacks.csv')) {
        if (($handle = fopen('attacks.csv', "r")) !== FALSE) {
            $headers = fgetcsv($handle); // ID,Type,Algorithm
            while (($data = fgetcsv($handle)) !== FALSE) {
                if(isset($data[0])) {
                    $attacks[$data[0]] = [
                        'Attack Type' => $data[1] ?? '',
                        'Algorithm' => $data[2] ?? ''
                    ];
                }
            }
            fclose($handle);
        }
    }

    $rows = [];
    if (($handle = fopen($base_csv, "r")) !== FALSE) {
        $headers = fgetcsv($handle); // Get headers
        while (($data = fgetcsv($handle)) !== FALSE) {
            $row = [];
            foreach ($headers as $index => $key) {
                // Remove BOM if present in first key
                if ($index === 0) {
                    $key = preg_replace('/[\x00-\x1F\x80-\xFF]/', '', $key);
                }
                $row[trim($key)] = $data[$index] ?? '';
            }
            
            // Enrich with attack info
            if (isset($row['Attack']) && isset($attacks[$row['Attack']])) {
                $row = array_merge($row, $attacks[$row['Attack']]);
            }
            
            $rows[] = $row;
        }
        fclose($handle);
    }
    echo json_encode($rows);
    exit;
}

if ($action === 'get_user') {
    $username = $_GET['username'] ?? '';
    if (!$username) {
        echo json_encode(['error' => 'No username provided']);
        exit;
    }
    // Sanitize username to prevent path traversal (allow unicode, dots, spaces)
    $username = preg_replace('/[^\p{L}\p{N}_.\s-]/u', '', $username);
    
    $file = $users_dir . $username . '.json';
    if (file_exists($file)) {
        echo file_get_contents($file);
    } else {
        echo json_encode((object)[]);
    }
    exit;
}

if ($action === 'save_user') {
    $input = json_decode(file_get_contents('php://input'), true);
    $username = $input['username'] ?? '';
    $data = $input['data'] ?? [];
    
    if (!$username) {
        echo json_encode(['error' => 'No username provided']);
        exit;
    }
    
    // Sanitize username to prevent path traversal (allow unicode, dots, spaces)
    $username = preg_replace('/[^\p{L}\p{N}_.\s-]/u', '', $username);

    $file = $users_dir . $username . '.json';
    file_put_contents($file, json_encode($data, JSON_PRETTY_PRINT));
    echo json_encode(['status' => 'success']);
    exit;
}

if ($action === 'create_user') {
    $input = json_decode(file_get_contents('php://input'), true);
    $username = $input['username'] ?? '';
    
    if (!$username) {
        echo json_encode(['error' => 'No username provided']);
        exit;
    }
    
    // Sanitize username to prevent path traversal (allow unicode, dots, spaces)
    $username = preg_replace('/[^\p{L}\p{N}_.\s-]/u', '', $username);
    
    $file = $users_dir . $username . '.json';
    if (!file_exists($file)) {
        file_put_contents($file, '{}');
        echo json_encode(['status' => 'success']);
    } else {
        echo json_encode(['error' => 'User already exists']);
    }
    exit;
}

if ($action === 'list_users') {
    $files = glob($users_dir . '*.json');
    $users = array_map(function($f) {
        return basename($f, '.json');
    }, $files);
    echo json_encode($users);
    exit;
}

echo json_encode(['error' => 'Invalid action']);
?>
