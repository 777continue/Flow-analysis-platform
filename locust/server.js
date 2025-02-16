const express = require('express');
// 创建一个新的 Express 应用实例，用于处理 HTTP 请求
const app = express();

app.get('/', (req, res) => {
    res.send('Welcome to example.com!');
});

app.get('/about', (req, res) => {
    res.send('This is the about page of example.com.');
});

const port = 8080;
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});