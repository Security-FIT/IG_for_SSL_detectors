<?php
// media.php - Serves files from outside the web root
// Usage: media.php?type=audio&file=filename.flac
//        media.php?type=image&file=filename.png

// Disable error reporting display to avoid corrupting binaries
ini_set('display_errors', 0);

$type = $_GET['type'] ?? '';
$file = $_GET['file'] ?? '';

// Basic security: sanitize filename
$file = basename($file);

function serveFileWithRange($path, $contentType) {
    if (!file_exists($path)) {
        http_response_code(404);
        echo "File not found";
        exit;
    }

    $filesize = filesize($path);
    
    // Simplification: Always serve the whole file. 
    // This avoids issues with PHP chunking loops and buffer flushing causing cracking.
    // For small audio files (speech commands), loading the whole file is efficient.
    
    http_response_code(200);
    header("Content-Type: $contentType");
    header("Content-Length: $filesize");
    header("Accept-Ranges: bytes"); // Advertise support, but we might just send 200 OK depending on implementation
    header("Content-Transfer-Encoding: binary");
    header("Cache-Control: no-cache"); 

    // Clear any previous output buffers
    while (ob_get_level()) ob_end_clean();
    
    readfile($path);
    exit;
}

if ($type === 'audio') {
    $path = './recordings_48k/' . $file;
    serveFileWithRange($path, 'audio/flac');
} elseif ($type === 'image') {
    $path = '../outputs/IG/' . $file;
    // Images are usually small enough to serve directly, but range support doesn't hurt
    serveFileWithRange($path, 'image/png');
} else {
    http_response_code(400);
    echo "Invalid type";
}
?>
