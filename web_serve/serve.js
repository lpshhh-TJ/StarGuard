console.log("serve.js 被执行了");
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));  // 提供静态文件服务（index.html等）

// 初始数据
let elderStatus = {
    x: 2.5,
    y: 1.0, // 离地高度
    z: 2.0,
    isFallen: false
};

// 六个基站的坐标
let anchors = [
    { x: 0.23, y: 0.3, z: 0 },     // 基站1
    { x: 6, y: 2.7, z: 0 },   // 基站2
    { x: 4.6, y: 0, z: 4.2 },   // 基站3
    { x: 0, y: 2.55, z: 4.2 },     // 基站4
    { x: 3.8, y: 1.8, z: 0.0 }, // 基站5 (中心)
    { x: 0.0, y: 1.1, z: 2.1 }    // 基站6 (中心底部)
];

// 房间尺寸
let roomSize = { l: 6, h: 3, w: 4.2 };

// 房间区域配置（x/z 范围）
let roomConfig = {
    living: { xMin: 0, xMax: 2.5, zMin: 0, zMax: 2, color: '#f43f5e', name: '客厅' },
    bedroom: { xMin: 2.5, xMax: 5, zMin: 0, zMax: 2, color: '#3b82f6', name: '卧室' },
    kitchen: { xMin: 2.5, xMax: 5, zMin: 2, zMax: 4, color: '#22c55e', name: '厨房' },
    bathroom: { xMin: 0, xMax: 2.5, zMin: 2, zMax: 4, color: '#a855f7', name: '卫生间' }
};

// 数据采集状态
let datasetCollection = {
    isCollectingDataset: false,           // 正在采集数据集
    currentCollectingCoordinate: null,    // 当前采集坐标
    isCollectingCurrentCoordinate: false, // 正在采集当前坐标
    currentCoordinateCompleted: false,    // 该坐标已完成
    completedCoordinates: [],             // 已完成的坐标列表
    totalCoordinates: 25,                 // 总坐标数
    currentCoordinateIndex: 0,            // 当前坐标索引 (0-24)
    coordinates: []                       // 所有坐标点
};

app.post('/update', (req, res) => {
    elderStatus = req.body;
    console.log("收到坐标:", elderStatus);
    res.send({ status: "ok" });
});

app.get('/status', (req, res) => {
    res.json(elderStatus);
});

// 获取基站坐标
app.get('/anchors', (req, res) => {
    res.json({ anchors: anchors });
});

// 更新基站坐标
app.post('/anchors', (req, res) => {
    if (req.body.anchors && Array.isArray(req.body.anchors)) {
        anchors = req.body.anchors;
        console.log("基站坐标已更新:", anchors);
    }
    res.json({ status: "ok", anchors: anchors });
});

// 获取房间尺寸
app.get('/roomSize', (req, res) => {
    res.json({ roomSize: roomSize });
});

// 更新房间尺寸
app.post('/roomSize', (req, res) => {
    if (req.body.roomSize) {
        roomSize = req.body.roomSize;
        console.log("房间尺寸已更新:", roomSize);
    }
    res.json({ status: "ok", roomSize: roomSize });
});

// 获取房间区域配置
app.get('/roomConfig', (req, res) => {
    res.json({ roomConfig: roomConfig });
});

// 更新房间区域配置
app.post('/roomConfig', (req, res) => {
    if (req.body.roomConfig && typeof req.body.roomConfig === 'object') {
        roomConfig = req.body.roomConfig;
        console.log("房间区域配置已更新:", roomConfig);
    }
    res.json({ status: "ok", roomConfig: roomConfig });
});

// ========== 数据采集 API ==========

// 生成 5x5 网格坐标（6段中的内点，不贴墙）
function generateGridCoordinates(roomSize, height) {
    const coordinates = [];
    const gridSize = 5;
    const segments = 6;  // 分成6段

    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            coordinates.push({
                x: (roomSize.l / segments) * (i + 1),  // 1/6, 2/6, 3/6, 4/6, 5/6
                y: height,
                z: (roomSize.w / segments) * (j + 1)   // 1/6, 2/6, 3/6, 4/6, 5/6
            });
        }
    }
    return coordinates;
}

// 设置当前采集坐标
function setCurrentCoordinate() {
    if (datasetCollection.currentCoordinateIndex < datasetCollection.coordinates.length) {
        datasetCollection.currentCollectingCoordinate =
            datasetCollection.coordinates[datasetCollection.currentCoordinateIndex];
        datasetCollection.currentCoordinateCompleted = false;
    } else {
        // 所有坐标已完成
        datasetCollection.isCollectingDataset = false;
        datasetCollection.currentCollectingCoordinate = null;
        console.log("数据集采集完成！共完成", datasetCollection.completedCoordinates.length, "个坐标");
    }
}

// 移动到下一个坐标
function moveToNextCoordinate() {
    datasetCollection.currentCoordinateIndex++;
    setCurrentCoordinate();
}

// 获取数据采集状态
app.get('/dataset-status', (req, res) => {
    res.json(datasetCollection);
});

// 获取当前采集坐标
app.get('/dataset-current-coordinate', (req, res) => {
    res.json({
        coordinate: datasetCollection.currentCollectingCoordinate,
        isCollecting: datasetCollection.isCollectingCurrentCoordinate,
        isCompleted: datasetCollection.currentCoordinateCompleted
    });
});

// 获取数据集采集状态（轻量级）
app.get('/dataset-is-collecting', (req, res) => {
    res.json({
        isCollectingDataset: datasetCollection.isCollectingDataset
    });
});

// 开始数据集采集
app.get('/dataset-start', (req, res) => {
    if (datasetCollection.isCollectingDataset) {
        return res.json({ status: "already_started" });
    }

    // 获取用户设置的采集高度，默认为 1.0
    const height = req.query.height !== undefined ? parseFloat(req.query.height) : 1.0;

    datasetCollection.coordinates = generateGridCoordinates(roomSize, height);
    datasetCollection.isCollectingDataset = true;
    datasetCollection.currentCoordinateIndex = 0;
    datasetCollection.completedCoordinates = [];
    setCurrentCoordinate();

    console.log("数据集采集已启动，共", datasetCollection.coordinates.length, "个坐标，采集高度:", height, "m");
    res.json({ status: "started", datasetCollection });
});

// 开始采集当前坐标
app.get('/dataset-coordinate/start', (req, res) => {
    if (!datasetCollection.isCollectingDataset) {
        return res.json({ status: "not_started" });
    }

    datasetCollection.isCollectingCurrentCoordinate = true;
    datasetCollection.currentCoordinateCompleted = false;

    console.log("开始采集坐标:", datasetCollection.currentCollectingCoordinate);
    res.json({ status: "collecting", coordinate: datasetCollection.currentCollectingCoordinate });
});

// 完成当前坐标（外部设备调用）
app.post('/dataset-coordinate/complete', (req, res) => {
    if (!datasetCollection.isCollectingCurrentCoordinate) {
        return res.json({ status: "not_collecting" });
    }

    datasetCollection.currentCoordinateCompleted = true;
    datasetCollection.completedCoordinates.push(datasetCollection.currentCollectingCoordinate);
    datasetCollection.isCollectingCurrentCoordinate = false;

    moveToNextCoordinate();

    console.log("坐标采集完成，移动到下一个坐标");
    res.json({ status: "completed", datasetCollection });
});

// 跳过当前坐标
app.post('/dataset-coordinate/skip', (req, res) => {
    if (!datasetCollection.isCollectingDataset) {
        return res.json({ status: "not_started" });
    }

    datasetCollection.isCollectingCurrentCoordinate = false;
    datasetCollection.currentCoordinateCompleted = false;

    moveToNextCoordinate();

    console.log("跳过当前坐标");
    res.json({ status: "skipped", datasetCollection });
});

// 停止数据集采集
app.post('/dataset-stop', (req, res) => {
    datasetCollection = {
        isCollectingDataset: false,
        currentCollectingCoordinate: null,
        isCollectingCurrentCoordinate: false,
        currentCoordinateCompleted: false,
        completedCoordinates: [],
        totalCoordinates: 25,
        currentCoordinateIndex: 0,
        coordinates: []
    };

    console.log("数据集采集已停止");
    res.json({ status: "stopped" });
});

app.listen(3000, () => console.log("服务已启动: localhost:3000"));
