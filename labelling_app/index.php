<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Attribution Labelling App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">

<?php include 'nav.php'; ?>

<div class="container mt-5">
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card shadow-sm">
                <div class="card-body p-5 text-center">
                    <h1 class="mb-3">Attribution Labelling</h1>
                    <p class="text-muted mb-5">
                        Tool for annotating Integrated Gradients attributions for AASIST, CAMHFA, and SLS detectors.
                    </p>

                    <div class="mb-4 text-start">
                        <label for="existingUsers" class="form-label">Select User</label>
                        <select id="existingUsers" class="form-select" onchange="enableStart()">
                            <!-- Populated by JS -->
                        </select>
                    </div>

                    <div class="d-grid gap-2 mb-4">
                        <button class="btn btn-primary" id="startBtn" onclick="startLabelling()">Start Labelling</button>
                    </div>
                    
                    <div class="mb-4 d-grid gap-2">
                        <a href="artefacts_eer.php" class="btn btn-outline-info">View Artefact EER Analysis</a>
                        <a href="methodology.php" class="btn btn-outline-secondary">View Methodology Instructions</a>
                    </div>

                    <hr>

                    <div class="mb-3 text-start">
                        <label for="newUsername" class="form-label">Create New User</label>
                        <div class="input-group">
                            <input type="text" id="newUsername" class="form-control" placeholder="Enter name">
                            <button class="btn btn-outline-success" type="button" onclick="createNewUser()">Create</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
    async function loadUsers() {
        const res = await fetch('api.php?action=list_users');
        const users = await res.json();
        const select = document.getElementById('existingUsers');
        select.innerHTML = '';
        users.forEach(u => {
            const opt = document.createElement('option');
            opt.value = u;
            opt.textContent = u;
            select.appendChild(opt);
        });
        
        // Default to "Vojta" if exists
        if(users.includes("Vojta")) {
            select.value = "Vojta";
        }
        enableStart();
    }

    function enableStart() {
        const select = document.getElementById('existingUsers');
        document.getElementById('startBtn').disabled = !select.value;
    }

    async function createNewUser() {
        const username = document.getElementById('newUsername').value.trim();
        if(!username) return;

        const res = await fetch('api.php?action=create_user', {
            method: 'POST',
            body: JSON.stringify({ username })
        });
        const result = await res.json();
        
        if (result.status === 'success') {
            await loadUsers();
            document.getElementById('existingUsers').value = username; // Select new user
            document.getElementById('newUsername').value = '';
        } else {
            alert(result.error || "Error creating user");
        }
    }

    function startLabelling() {
        const user = document.getElementById('existingUsers').value;
        if(user) {
            window.location.href = `annotate.php?user=${encodeURIComponent(user)}`;
        }
    }

    window.onload = loadUsers;
</script>

</body>
</html>
