const express = require('express');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Ensure sites directory exists
const sitesDir = path.join(__dirname, 'sites');
if (!fs.existsSync(sitesDir)) {
    fs.mkdirSync(sitesDir, { recursive: true });
}

// Serve main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// API endpoint to create a new website
app.post('/api/create-site', (req, res) => {
    try {
        const { title, content } = req.body;
        
        if (!title || !content) {
            return res.status(400).json({ error: 'Title and content are required' });
        }

        // Generate unique site ID
        const siteId = uuidv4();
        const siteDir = path.join(sitesDir, siteId);
        
        // Create site directory
        fs.mkdirSync(siteDir, { recursive: true });
        
        // Create basic HTML file for the site
        const htmlContent = `<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .content {
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <h1>${title}</h1>
    <div class="content">
        ${content.replace(/\n/g, '<br>')}
    </div>
</body>
</html>`;
        
        // Write the HTML file
        fs.writeFileSync(path.join(siteDir, 'index.html'), htmlContent);
        
        // Return the site URL
        const siteUrl = `${req.protocol}://${req.get('Host')}/site/${siteId}`;
        res.json({ 
            success: true, 
            siteId, 
            url: siteUrl,
            message: 'Сайт успешно создан!'
        });
        
    } catch (error) {
        console.error('Error creating site:', error);
        res.status(500).json({ error: 'Ошибка при создании сайта' });
    }
});

// Serve created websites
app.use('/site/:siteId', (req, res, next) => {
    const siteId = req.params.siteId;
    const sitePath = path.join(sitesDir, siteId);
    
    if (!fs.existsSync(sitePath)) {
        return res.status(404).send('Сайт не найден');
    }
    
    express.static(sitePath)(req, res, next);
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});