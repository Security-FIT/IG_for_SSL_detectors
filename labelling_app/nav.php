<?php
$current_page = basename($_SERVER['PHP_SELF']);
?>
<nav class="navbar navbar-expand-lg navbar-dark bg-dark mb-0 flex-shrink-0">
    <div class="container-fluid">
        <a class="navbar-brand" href="index.php">Attribution Labelling</a>
        <div class="navbar-nav me-auto">
             <a class="nav-link <?php echo $current_page == 'index.php' ? 'active' : ''; ?>" href="index.php">Home</a>
             <a class="nav-link <?php echo $current_page == 'methodology.php' ? 'active' : ''; ?>" href="methodology.php" target="_blank">Methodology</a>
             <a class="nav-link <?php echo $current_page == 'artefacts_eer.php' ? 'active' : ''; ?>" href="artefacts_eer.php" target="_blank">Artefact Stats</a>
        </div>
        <span class="navbar-text text-light">
            User: <span id="usernameVal" class="fw-bold text-info"></span>
        </span>
    </div>
</nav>
